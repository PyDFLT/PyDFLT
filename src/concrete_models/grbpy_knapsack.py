import cvxpy as cp
import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel

from src.abstract_models.grbpy import GRBPYModel


class GRBPYKnapsackModel(GRBPYModel, optGrbModel):
    def __init__(
        self,
        num_decisions: int,
        capacity: float,
        weights_lb: float = 3.0,
        weights_ub: float = 8.0,
        dimension: int = 1,
        seed: int = None,
        num_scenarios: int = 1,
    ):
        """
        Creates a GRBPYKnapsackModel instance

        Args:
            num_decisions (int): Number of decision variables/items for the knapsack
            capacity (float): Capacity of the knapsack
            weights_lb (float): Lower bound of the items weights, to generate item weights
            weights_ub (float): Upper bound of the items weights, to generate item weights
            dimension (int): Dimension of weights of the items
            seed (int): Seed (influences weight generation)
            num_scenarios (int): Number of scenarios the problem consists of
        """
        # Setting input parameters
        self.num_decisions = num_decisions
        self.capacity = capacity
        self.weights_lb = weights_lb
        self.weights_ub = weights_ub
        self.dimension = dimension
        self.seed = seed
        self.num_scenarios = num_scenarios

        # Setting basic model parameters
        model_sense = "MAX"
        var_shapes = {"select_item": (num_decisions,)}  # x: take item decision
        _shape = (num_decisions, num_scenarios) if num_scenarios > 1 else (num_decisions,)
        param_to_predict_shapes = {"item_value": _shape}
        extra_param_shapes = None

        # Setting additional model parameters
        np.random.seed(seed)

        # Initialize fixed parameters
        self.weights = np.random.uniform(weights_lb, weights_ub, (dimension, num_decisions))
        self.capacity_np = self.capacity * np.ones(dimension)

        GRBPYModel.__init__(
            self,
            var_shapes,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=extra_param_shapes,
        )

    def _create_model(self):
        """Creates model and vars_dict"""
        # Create a GP model
        self.gp_model = gp.Model("knapsack")
        self.vars_dict = {}

        # Define vars
        name = "select_item"
        x = self.gp_model.addMVar(self.num_vars, name=name, vtype=GRB.BINARY)
        self.vars_dict[name] = x
        # It is a maximization problem
        self.gp_model.modelSense = self.modelSense = self.model_sense_int

        # Constraints
        self.gp_model.addConstr(self.weights @ x <= self.capacity_np)

        # To make sure pyepo losses work
        self.x = x
        self._model = self.gp_model

    def _set_params(self, *params_i: np.ndarray):
        (c_i,) = params_i
        if self.num_scenarios > 1:
            self.gp_model.setObjective(gp.quicksum(self.vars_dict["select_item"] @ np.array(c_i)))
        else:
            self.gp_model.setObjective(self.vars_dict["select_item"] @ np.array(c_i))

    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] = None,
    ) -> torch.float:
        return (data_batch["item_value"] * decisions_batch["select_item"]).sum(-1)

    def _getModel(
        self,
    ):  # This function overrides the _getModel from the optGrbModel class
        self._create_model()
        return self.gp_model, self.vars_dict["select_item"]

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        # Since this function is to create a quadratic variant, we only care about first stage variables
        var_domain_dict = {"select_item": {"boolean": True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    def get_constraints(self, vars_dict: dict[str, cp.Variable]):
        # These constraints are cvxpy style constraints
        x = vars_dict["select_item"]
        return [self.weights @ x <= self.capacity_np]
