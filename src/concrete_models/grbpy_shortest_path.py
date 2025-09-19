import cvxpy as cp
import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel

from src.abstract_models.grbpy import GRBPYModel


class ShortestPath(GRBPYModel, optGrbModel):
    """
    A Gurobi-based shortest path optimization model.
    This model solves a shortest path problem on a grid network where the goal is to find
    the path with minimum cost from the source to the sink.

    Attributes:
        grid (tuple[int, int]): The dimensions of the grid network.
        num_scenarios (int): Number of scenarios for multi-scenario optimization.
        arcs (list): List of arcs (edges) in the grid network.
        arcs_to_index (dict): Mapping from arcs to their indices.
    """

    def __init__(
        self,
        grid: tuple[int, int],
        num_scenarios: int = 1,
    ):
        """
        Initializes the ShortestPath model.

        Args:
            grid (tuple[int, int]): The dimensions of the grid network (rows, columns).
            num_scenarios (int): Number of scenarios for multi-scenario optimization. Defaults to 1.
        """
        # Setting input parameters
        self.grid = grid
        self.num_scenarios = num_scenarios

        # Setting basic model parameters
        model_sense = "MIN"
        num_coefficients = grid[0] * (grid[1] - 1) + grid[1] * (grid[0] - 1)
        _shape = (num_coefficients, num_scenarios) if num_scenarios > 1 else (num_coefficients,)
        param_to_predict_shapes = {"arc_costs": _shape}
        extra_param_shapes = None

        # Setting additional model parameters
        self.arcs = self._get_arcs()
        self.arcs_to_index = {arc: i for i, arc in enumerate(self.arcs)}
        var_shapes = {"select_arc": (len(self.arcs),)}

        GRBPYModel.__init__(
            self,
            var_shapes,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=extra_param_shapes,
        )

    def _create_model(self):
        """
        Creates the Gurobi optimization model for the shortest path problem.
        This method defines the decision variables, flow conservation constraints, and model sense.
        """
        """Creates model AND vars_dict dimension"""
        # Create a GP model
        gp_model = gp.Model("shortestpath")
        vars_dict = {}

        # Define vars
        name = "select_arc"
        x = gp_model.addMVar((len(self.arcs),), name=name, vtype=GRB.BINARY)
        vars_dict[name] = x
        # Set model sense
        gp_model.modelSense = self.modelSense = self.model_sense_int

        # Constraints
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[self.arcs_to_index[e]]
                    # flow out
                    elif v == e[0]:
                        expr -= x[self.arcs_to_index[e]]
                # source
                if i == 0 and j == 0:
                    gp_model.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    gp_model.addConstr(expr == 1)
                # transition
                else:
                    gp_model.addConstr(expr == 0)

        return gp_model, vars_dict

    def _get_arcs(self):
        """
        Generates the list of arcs (edges) for the grid network.
        Source: PyEPO

        Returns:
            list: A list of tuples representing arcs (source_node, destination_node).
        """
        arcs = []
        for i in range(self.grid[0]):
            # edges on rows
            for j in range(self.grid[1] - 1):
                v = i * self.grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == self.grid[0] - 1:
                continue
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                arcs.append((v, v + self.grid[1]))
        return arcs

    def _set_params(self, *params_i: np.ndarray):
        """
        Sets the parameters for the shortest path model for a single instance.
        Updates the objective function with the provided arc costs.

        Args:
            *params_i (np.ndarray): Arc costs for the current instance.
        """
        (c_i,) = params_i
        if self.num_scenarios > 1:
            self.gp_model.setObjective(gp.quicksum(self.vars_dict["select_arc"] @ np.array(c_i)))
        else:
            self.gp_model.setObjective(self.vars_dict["select_arc"] @ np.array(c_i))

    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] = None,
    ) -> torch.float:
        """
        Computes the objective function value for the shortest path problem.
        The objective is to minimize the total cost of selected arcs.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data, including 'arc_costs'.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing decision variables, including 'select_arc'.
            predictions_batch (dict[str, torch.Tensor], optional): Unused for this implementation. Defaults to None.

        Returns:
            torch.float: The total cost of selected arcs for the batch.
        """
        return (data_batch["arc_costs"] * decisions_batch["select_arc"]).sum(-1)

    def _getModel(
        self,
    ):
        """
        Returns the Gurobi model and decision variables.
        This function overrides the _getModel from the optGrbModel class.

        Returns:
            tuple: A tuple containing the Gurobi model and the decision variables.
        """
        self._create_model()
        return self.gp_model, self.vars_dict["select_arc"]

    def get_constraints(self, vars_dict: dict[str, cp.Variable]):
        """
        Returns the constraints for the shortest path problem in CVXPY format.
        Used when creating a quadratic variant. These are flow conservation constraints.

        Args:
            vars_dict (dict[str, cp.Variable]): A dictionary mapping variable names to CVXPY variables.

        Returns:
            list: A list of CVXPY constraints for the shortest path problem.
        """
        x = vars_dict["select_arc"]
        constraints = []
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                v = i * self.grid[1] + j
                expr = 0
                for e in self.arcs:
                    # flow in
                    if v == e[1]:
                        expr += x[self.arcs_to_index[e]]
                    # flow out
                    elif v == e[0]:
                        expr -= x[self.arcs_to_index[e]]
                # source
                if i == 0 and j == 0:
                    constraints.append(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[1] - 1:
                    constraints.append(expr == 1)
                # transition
                else:
                    constraints.append(expr == 0)
        return constraints
