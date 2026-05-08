from itertools import combinations
from typing import Any

import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pydflt.abstract_models.base import MIN
from pydflt.abstract_models.grbpy_two_stage import GRBPYTwoStageModel


class TwoStageProbabilisticTSP(GRBPYTwoStageModel):
    """
    A Gurobi-based two-stage probabilistic Traveling Salesman Problem (TSP) model.

    Cities are arranged approximately on a circle (with Gaussian radial noise) around
    a central depot. In the first stage the planner decides which cities to include in
    a tour and which cities to serve via individual direct round-trips. In the second
    stage, uncertain visit requirements are revealed per scenario: cities that do not
    need to be visited can have their direct-trip canceled (yielding a partial cost
    recovery), while cities that need a visit but were not planned incur a missed-city
    penalty.

    Subtour elimination is handled via Gurobi lazy constraints.

    Attributes:
        num_cities (int): Number of cities (excluding the depot).
        missed_city_penalty (float): Penalty multiplier applied when a required city
            is not visited.
        recovery_ratio (float): Fraction of the direct-trip cost recovered when a
            scheduled direct trip is canceled because the city does not need a visit.
        radius (float): Base radius of the circular city layout.
        noise_std (float): Standard deviation of the radial noise applied to city
            positions.
        seed (int): Random seed for reproducible coordinate generation.
        num_scenarios (int): Number of second-stage scenarios.
        num_nodes (int): Total number of nodes (cities + depot).
        x_coord (np.ndarray): X coordinates of all nodes (depot first).
        y_coord (np.ndarray): Y coordinates of all nodes (depot first).
        distances (np.ndarray): Pairwise Euclidean distance matrix of shape
            ``(num_nodes, num_nodes)``.
    """

    def __init__(
        self,
        num_cities: int,
        missed_city_penalty: float,
        recovery_ratio: float,
        radius: float = 10,
        noise_std: float = 1,
        seed: int = 5,
        num_scenarios: int = 1,
    ):
        """
        Initializes the TwoStageProbabilisticTSP model.

        Args:
            num_cities (int): Number of cities (excluding the depot).
            missed_city_penalty (float): Penalty multiplier for unvisited required cities.
            recovery_ratio (float): Fraction of direct-trip cost recovered on cancellation.
            radius (float): Base radius of the circular city layout. Defaults to 10.
            noise_std (float): Standard deviation of radial noise. Defaults to 1.
            seed (int): Random seed for coordinate generation. Defaults to 5.
            num_scenarios (int): Number of second-stage scenarios. Defaults to 1.
        """
        self.num_cities = num_cities
        self.missed_city_penalty = missed_city_penalty
        self.recovery_ratio = recovery_ratio
        self.radius = radius
        self.noise_std = noise_std
        self.seed = seed
        self.num_scenarios = num_scenarios

        self.rng = np.random.default_rng(self.seed)
        self.num_nodes = self.num_cities + 1
        self.x_coord, self.y_coord = self._get_coords()
        self.distances = self._determine_distances()

        model_sense = MIN
        decision_variables = {
            "x_arc": (self.num_nodes, self.num_nodes),
            "x_direct": (self.num_cities,),
        }
        _shape = (self.num_cities, num_scenarios) if num_scenarios > 1 else (num_cities,)
        param_to_predict_shapes = {"visit": _shape}

        GRBPYTwoStageModel.__init__(
            self,
            decision_variables,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=None,
        )

        self.lazy_constraints_method = self.subtourelim
        self.gp_model.Params.lazyConstraints = 1

    def _create_model(self) -> tuple[gp.Model, dict[str, gp.MVar | gp.Var]]:
        """
        Creates the Gurobi optimization model for the two-stage probabilistic TSP.

        Defines first-stage variables (arc traversal and direct trips), auxiliary
        variables (city visited indicator and tour-existence flag), and second-stage
        variables (city cancellation per scenario). Structural constraints (symmetry,
        degree, tour-existence linking) are added here; scenario-dependent constraints
        and the objective are set in ``_set_params``.

        Returns:
            tuple: A tuple ``(gp_model, vars_dict)`` where ``vars_dict`` contains only
                the first-stage decision variables keyed by name.
        """
        gp_model = gp.Model("probabilistic_travelling_salesperson")
        vars_dict = {}
        second_stage_vars_dict = {}
        auxiliary_vars_dict = {}

        arc_traversed = gp_model.addMVar((self.num_nodes, self.num_nodes), name="x_arc", vtype=GRB.BINARY)
        direct_trip = gp_model.addMVar((self.num_cities,), name="x_direct", vtype=GRB.BINARY)
        city_visited = gp_model.addMVar((self.num_cities,), name="x_visited", vtype=GRB.BINARY)
        tour_exists = gp_model.addMVar((1,), name="x_tour", vtype=GRB.BINARY)
        city_canceled = gp_model.addMVar((self.num_cities, self.num_scenarios), name="y_canceled", vtype=GRB.BINARY)

        # Enforce arc symmetry and no self-loops
        gp_model.addConstrs(arc_traversed[i, j] == arc_traversed[j, i] for i in range(self.num_nodes) for j in range(i + 1, self.num_nodes))
        gp_model.addConstrs(arc_traversed[i, i] == 0 for i in range(self.num_nodes))

        # A direct trip implies the city is visited
        gp_model.addConstrs(city_visited[i] >= direct_trip[i] for i in range(self.num_cities))

        # Degree-2 constraint for cities visited via the tour (not direct trips)
        gp_model.addConstrs(
            gp.quicksum(arc_traversed[i + 1, j] for j in range(self.num_nodes)) == 2 * (city_visited[i] - direct_trip[i]) for i in range(self.num_cities)
        )
        # Depot has degree 2 iff a tour exists
        gp_model.addConstr(gp.quicksum(arc_traversed[0, j] for j in range(self.num_nodes)) == 2 * tour_exists[0])

        # Tour exists iff at least one city is visited via the tour
        gp_model.addConstr(tour_exists[0] >= gp.quicksum(city_visited[i] - direct_trip[i] for i in range(self.num_cities)) / self.num_cities)

        vars_dict["x_arc"] = arc_traversed
        vars_dict["x_direct"] = direct_trip
        auxiliary_vars_dict["x_visited"] = city_visited
        auxiliary_vars_dict["x_tour"] = tour_exists
        second_stage_vars_dict["y_canceled"] = city_canceled

        gp_model.modelSense = GRB.MINIMIZE
        assert self.model_sense_int == gp_model.modelSense, "Is it a maximization or minimization problem? Check model sense."

        self.second_stage_vars_dict = second_stage_vars_dict
        self.auxiliary_vars_dict = auxiliary_vars_dict

        return gp_model, vars_dict

    def _set_params(self, *parameters_i: np.ndarray) -> None:
        """
        Sets scenario-dependent constraints and the objective for a single instance.

        Removes any previously added cancellation constraints, then adds new ones based
        on the realized ``visit`` requirements and sets the full objective.

        Args:
            *parameters_i (np.ndarray): Realized visit requirements of shape
                ``(num_cities,)`` (single scenario) or ``(num_cities * num_scenarios,)``
                (multi-scenario, will be reshaped).
        """
        requires_visit = parameters_i[0]
        requires_visit = requires_visit.reshape(-1, self.num_scenarios)
        requires_visit = np.round(np.clip(requires_visit, 0, 1))

        arc_traversed = self.vars_dict["x_arc"]
        direct_trip = self.vars_dict["x_direct"]
        city_visited = self.auxiliary_vars_dict["x_visited"]
        city_canceled = self.second_stage_vars_dict["y_canceled"]

        # Remove existing cancellation constraints
        if len(self.gp_model.getConstrs()):
            constraints_to_remove = []
            for k in range(self.num_scenarios):
                for i in range(self.num_cities):
                    c = self.gp_model.getConstrByName(f"canceled[{i},{k}]")
                    if c is not None:
                        constraints_to_remove.append(c)
            for c in constraints_to_remove:
                self.gp_model.remove(c)

        # Can only cancel a direct trip if the city does not need to be visited
        self.gp_model.addConstrs(
            (city_canceled[i, k] <= direct_trip[i] * (1 - int(requires_visit[i, k])) for i in range(self.num_cities) for k in range(self.num_scenarios)),
            name="canceled",
        )

        obj = (
            gp.quicksum(self.distances[i, j] * arc_traversed[i, j] for i in range(self.num_nodes) for j in range(i + 1, self.num_nodes))
            + gp.quicksum(2 * self.distances[0, i + 1] * direct_trip[i] for i in range(self.num_cities))
            + (1 / self.num_scenarios)
            * gp.quicksum(
                requires_visit[i, k] * (1 - city_visited[i]) * self.missed_city_penalty * 2 * self.distances[0, i + 1]
                for i in range(self.num_cities)
                for k in range(self.num_scenarios)
            )
            - (self.recovery_ratio / self.num_scenarios)
            * gp.quicksum(2 * self.distances[0, i + 1] * city_canceled[i, k] for i in range(self.num_cities) for k in range(self.num_scenarios))
        )
        self.gp_model.setObjective(obj)

    def _get_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates node coordinates: cities on a noisy circle, depot at the origin.

        Returns:
            tuple: ``(x_coord, y_coord)`` arrays of length ``num_nodes``, with the
                depot at index 0.
        """
        angles = np.linspace(0, 2 * np.pi, self.num_cities, endpoint=False)
        noise = self.rng.normal(0, self.noise_std, self.num_cities)
        perturbed_x = (self.radius + noise) * np.cos(angles)
        perturbed_y = (self.radius + noise) * np.sin(angles)
        x_coord = np.insert(perturbed_x, 0, 0.0)
        y_coord = np.insert(perturbed_y, 0, 0.0)
        return x_coord, y_coord

    def _determine_distances(self) -> np.ndarray:
        """
        Computes the pairwise Euclidean distance matrix for all nodes.

        Returns:
            np.ndarray: Distance matrix of shape ``(num_nodes, num_nodes)``.
        """
        distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                distances[i, j] = np.sqrt((self.x_coord[i] - self.x_coord[j]) ** 2 + (self.y_coord[i] - self.y_coord[j]) ** 2)
        return distances

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        """
        Returns the variable domains for the first-stage variables used when
        creating a CVXPY quadratic variant.

        Returns:
            dict[str, dict[str, bool]]: Domain specifications for ``x_arc`` and
                ``x_direct`` (both binary).
        """
        return {
            "x_arc": {"boolean": True},
            "x_direct": {"boolean": True},
        }

    @staticmethod
    def get_constraints(vars_dict: dict[str, cp.Variable]) -> list[Any]:
        """
        Returns CVXPY constraints for the quadratic variant.

        No additional constraints beyond variable domains are required for the
        quadratic proxy.

        Args:
            vars_dict (dict[str, cp.Variable]): First-stage CVXPY variables.

        Returns:
            list: Empty list.
        """
        return []

    @staticmethod
    def subtourelim(model, where):
        """
        Gurobi lazy-constraint callback for subtour elimination.

        Finds the shortest cycle in the current MIP solution and, if it is shorter
        than the full tour, adds a subtour-elimination constraint via ``cbLazy``.

        Args:
            model: The Gurobi model passed by the callback mechanism.
            where: The Gurobi callback location code.
        """
        if where == GRB.Callback.MIPSOL:
            arc_vars = [var for var in model.getVars() if "x_arc" in var.VarName]
            vals = model.cbGetSolution(arc_vars)
            selected = gp.tuplelist((var.VarName[-4], var.VarName[-2]) for i, var in enumerate(arc_vars) if vals[i] > 0.5)

            direct_vars = [var for var in model.getVars() if "x_direct" in var.VarName]
            direct_vals = model.cbGetSolution(direct_vars)
            direct_trips = sum(direct_vals[i] for i in range(len(direct_vals)))
            visited_vars = [var for var in model.getVars() if "x_visited" in var.VarName]
            visited_vals = model.cbGetSolution(visited_vars)
            visited_cities = sum(visited_vals[i] for i in range(len(visited_vals)))
            to_visit_with_tour = visited_cities - direct_trips

            unvisited = [i + 1 for i, value in enumerate(visited_vals) if value > 0.5]
            tour = [i + 1 for i, value in enumerate(visited_vals) if value > 0.5]
            unvisited.insert(0, 0)
            tour.insert(0, 0)
            while unvisited:
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in selected.select(current, "*") if j in unvisited]
                if len(thiscycle) <= len(tour):
                    tour = thiscycle

            if len(tour) < to_visit_with_tour:
                model.cbLazy(gp.quicksum(arc_vars[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)
