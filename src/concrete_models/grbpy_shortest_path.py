import cvxpy as cp
import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel

from src.abstract_models.grbpy import GRBPYModel


class ShortestPath(GRBPYModel, optGrbModel):
    def __init__(
        self,
        grid: tuple[int, int],
        num_scenarios: int = 1,
    ):
        # Setting input parameters
        self.grid = grid
        self.num_scenarios = num_scenarios

        # Setting basic model parameters
        model_sense = "MIN"
        num_coefficients = grid[0] * (grid[1] - 1) + grid[1] * (grid[0] - 1)
        _shape = (num_coefficients, num_scenarios) if num_scenarios > 1 else (num_coefficients,)
        param_to_predict_shapes = {"c": _shape}
        extra_param_shapes = None

        # Setting additional model parameters
        self.arcs = self._get_arcs()
        self.arcs_to_index = {arc: i for i, arc in enumerate(self.arcs)}
        var_shapes = {"x": (len(self.arcs),)}

        GRBPYModel.__init__(
            self,
            var_shapes,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=extra_param_shapes,
        )

    def _create_model(self):
        """Creates model AND vars_dict dimension"""
        # Create a GP model
        self.gp_model = gp.Model("knapsack")
        self.vars_dict = {}

        # Define vars
        name = "x"
        x = self.gp_model.addMVar((len(self.arcs),), name=name, vtype=GRB.BINARY)
        self.vars_dict[name] = x
        # Set model sense
        self.gp_model.modelSense = self.modelSense = self.model_sense_int

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
                    self.gp_model.addConstr(expr == -1)
                # sink
                elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                    self.gp_model.addConstr(expr == 1)
                # transition
                else:
                    self.gp_model.addConstr(expr == 0)

        # To make sure pyepo losses work
        self.x = x
        self._model = self.gp_model

    def _get_arcs(self):
        """
        A method to get list of arcs for grid network
        Source: PyEPO
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
        (c_i,) = params_i
        if self.num_scenarios > 1:
            self.gp_model.setObjective(gp.quicksum(self.vars_dict["x"] @ np.array(c_i)))
        else:
            self.gp_model.setObjective(self.vars_dict["x"] @ np.array(c_i))

    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] = None,
    ) -> torch.float:
        return (data_batch["c"] * decisions_batch["x"]).sum(-1)

    def _getModel(
        self,
    ):  # This function overrides the _getModel from the optGrbModel class
        self._create_model()
        return self.gp_model, self.vars_dict["x"]

    def get_constraints(self, vars_dict: dict[str, cp.Variable]):
        # These constraints are cvxpy style constraints
        x = vars_dict["x"]
        return [self.weights @ x <= self.capacity_np]
