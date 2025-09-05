import random
from itertools import combinations

import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.abstract_models.base import MIN
from src.abstract_models.grbpy_two_stage import GRBPYTwoStageModel


class TwoStageProbabilisticVehicleRouting(GRBPYTwoStageModel):
    def __init__(
        self,
        num_cities: int,
        missed_city_penalty: float,
        recovery_ratio: float,
        single_city_trip_ratio: float = 1,
        radius: float = 10,
        noise_std: float = 1,
        seed: int = 5,
        num_scenarios: int = 1,
    ):
        # Getting parameters
        self.num_cities = num_cities
        self.missed_city_penalty = missed_city_penalty
        self.single_city_trip_ratio = single_city_trip_ratio
        self.radius = radius
        self.noise_std = noise_std
        self.recovery_ratio = recovery_ratio
        self.seed = seed
        self.num_scenarios = num_scenarios

        random.seed(self.seed)
        np.random.seed(self.seed)
        self.num_nodes = self.num_cities + 1
        self.x_coord, self.y_coord = self._get_coords()
        self.distances = self._determine_distances()

        # Setting basic model parameters
        model_sense = MIN
        decision_variables = {
            "x_arc": (self.num_nodes, self.num_nodes),
            "x_direct": (self.num_cities,),
            #'x_visited': (self.num_cities,),
            #'x_tour': (1, )
        }
        # num_decisions = self.num_cities * (self.num_cities + self.num_vehicle_types + 2) + self.num_vehicle_types

        if num_scenarios > 1:
            varying_parameters = {"visit": (self.num_cities, num_scenarios)}
        else:
            varying_parameters = {"visit": (self.num_cities,)}
        params_to_predict = [
            "visit",
        ]

        GRBPYTwoStageModel.__init__(self, decision_variables, varying_parameters, params_to_predict, model_sense)

        # We use lazy constraints
        self.lazy_constraints_method = self.subtourelim
        self.gp_model.Params.lazyConstraints = 1

    def _create_model(self):
        """Creates model AND variables_dict"""
        # Create a GP model
        self.gp_model = gp.Model("probabilistic_travelling_salesperson")
        self.vars_dict = {}
        self.second_stage_vars_dict = {}
        self.auxiliary_vars_dict = {}

        arc_traversed = self.gp_model.addMVar((self.num_nodes, self.num_nodes), name="x_arc", vtype=GRB.BINARY)
        direct_trip = self.gp_model.addMVar((self.num_cities,), name="x_direct", vtype=GRB.BINARY)
        city_visited = self.gp_model.addMVar((self.num_cities,), name="x_visited", vtype=GRB.BINARY)
        tour_exists = self.gp_model.addMVar((1,), name="x_tour", vtype=GRB.BINARY)
        city_canceled = self.gp_model.addMVar((self.num_cities, self.num_scenarios), name="y_canceled", vtype=GRB.BINARY)

        # Enforce symmetry
        self.gp_model.addConstrs(arc_traversed[i, j] == arc_traversed[j, i] for i in range(self.num_nodes) for j in range(i + 1, self.num_nodes))
        self.gp_model.addConstrs(arc_traversed[i, i] == 0 for i in range(self.num_nodes))

        # When there is a direct trip, the city is visited
        self.gp_model.addConstrs(city_visited[i] >= direct_trip[i] for i in range(self.num_cities))

        # Constraints: two edges incident to each city that is visited without a direct trip
        # self.gp_model.addConstrs(arc_traversed.sum(i + 1, '*') == 2*(city_visited[i] - direct_trip[i]) for i in range(self.num_cities))
        # self.gp_model.addConstrs(arc_traversed.sum(0, '*') == 2)
        self.gp_model.addConstrs(
            gp.quicksum(arc_traversed[i + 1, j] for j in range(self.num_nodes)) == 2 * (city_visited[i] - direct_trip[i]) for i in range(self.num_cities)
        )
        # Depot always has 2 edges if there is a tour
        self.gp_model.addConstr(gp.quicksum(arc_traversed[0, j] for j in range(self.num_nodes)) == 2 * tour_exists[0])

        # Tour exists if there are cities visited without direct trip (using indicator constraint
        self.gp_model.addConstr(tour_exists[0] >= gp.quicksum(city_visited[i] - direct_trip[i] for i in range(self.num_cities)) / self.num_cities)

        # Note: Use formulation by gurobi and as in pyepo. Make sure that skipped customers and direct_trip customers are not included in the presented edges
        self.vars_dict["x_arc"] = arc_traversed
        self.vars_dict["x_direct"] = direct_trip
        self.auxiliary_vars_dict["x_visited"] = city_visited
        self.auxiliary_vars_dict["x_tour"] = tour_exists
        self.second_stage_vars_dict["y_canceled"] = city_canceled

        self.gp_model.modelSense = self.model_sense

    def _set_params(self, *parameters_i: np.ndarray):
        """
        We get demands realized and add a constraint such that we serve all cities
        """
        # Obtain the weight parameters
        requires_visit = parameters_i[0]

        # Reshape the weights parameters
        requires_visit = requires_visit.reshape(-1, self.num_scenarios)
        requires_visit = np.round(np.clip(requires_visit, 0, 1))

        arc_traversed = self.vars_dict["x_arc"]
        direct_trip = self.vars_dict["x_direct"]
        city_visited = self.auxiliary_vars_dict["x_visited"]
        city_canceled = self.second_stage_vars_dict["y_canceled"]

        # Remove existing constraints
        if len(self.gp_model.getConstrs()):
            constraints_to_remove = []
            for scenario in range(self.num_scenarios):
                for city in range(self.num_cities):
                    canceled_constraint = self.gp_model.getConstrByName(f"canceled[{city},{scenario}]")
                    if canceled_constraint is not None:
                        constraints_to_remove.append(canceled_constraint)
            for constraint in constraints_to_remove:
                self.gp_model.remove(constraint)

        # y_missed becomes demand missed, this is smaller than demand (but at least one to keep flow)
        # Add constraints
        # Can only cancel if city doesn't have to be visited and there is a direct trip
        self.gp_model.addConstrs(
            (city_canceled[i, k] <= direct_trip[i] * (1 - int(requires_visit[i, k])) for i in range(self.num_cities) for k in range(self.num_scenarios)),
            name="canceled",
        )

        # Set objective
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

    def _get_coords(self):
        angles = np.linspace(0, 2 * np.pi, self.num_cities, endpoint=False)  # Equally spaced angles

        # Generate radial noise
        noise = np.random.normal(0, self.noise_std, self.num_cities)

        # Compute perturbed points
        perturbed_x = (self.radius + noise) * np.cos(angles)
        perturbed_y = (self.radius + noise) * np.sin(angles)

        # Add the depot
        x_coord = np.insert(perturbed_x, 0, 0.0)
        y_coord = np.insert(perturbed_y, 0, 0.0)

        return x_coord, y_coord

    def _determine_distances(self):
        distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                distances[i, j] = np.sqrt((self.x_coord[i] - self.x_coord[j]) ** 2 + (self.y_coord[i] - self.y_coord[j]) ** 2)

        return distances

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        # Since this function is to create a quadratic variant, we only care about first stage variables
        var_domain_dict = {
            "x_arc": {"boolean": True},
            "x_direct": {"boolean": True},
        }  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    @staticmethod
    def get_constraints(vars_dict: dict[str, cp.Variable]):
        # These constraints are cvxpy style constraints
        return []

    @staticmethod
    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            arc_vars = [var for var in model.getVars() if "x_arc" in var.VarName]
            vals = model.cbGetSolution(arc_vars)
            selected = gp.tuplelist((var.VarName[-4], var.VarName[-2]) for i, var in enumerate(arc_vars) if vals[i] > 0.5)

            # check how many direct tours there are and which vities are visited
            direct_vars = [var for var in model.getVars() if "x_direct" in var.VarName]
            direct_vals = model.cbGetSolution(direct_vars)
            direct_trips = sum(direct_vals[i] for i in range(len(direct_vals)))
            visited_vars = [var for var in model.getVars() if "x_visited" in var.VarName]
            visited_vals = model.cbGetSolution(visited_vars)
            visited_cities = sum(visited_vals[i] for i in range(len(visited_vals)))
            to_visit_with_tour = visited_cities - direct_trips

            # find the shortest cycle in the selected edge list
            unvisited = [i + 1 for i, value in enumerate(visited_vals) if value > 0.5]  # capitals[:]  #TODO: Probably not the right format
            tour = [i + 1 for i, value in enumerate(visited_vals) if value > 0.5]  # capitals[:]  # Dummy - guaranteed to be replaced
            unvisited.insert(0, 0)  # add depot
            tour.insert(0, 0)
            while unvisited:  # true if list is non-empty
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in selected.select(current, "*") if j in unvisited]
                if len(thiscycle) <= len(tour):
                    tour = thiscycle  # New shortest subtour

            if len(tour) < to_visit_with_tour:
                # add subtour elimination constr. for every pair of cities in subtour
                model.cbLazy(gp.quicksum(arc_vars[i, j] for i, j in combinations(tour, 2)) <= len(tour) - 1)
