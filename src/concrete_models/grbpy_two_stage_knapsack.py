import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.abstract_models.grbpy_two_stage import GRBPYTwoStageModel


class TwoStageKnapsack(GRBPYTwoStageModel):
    def __init__(
        self,
        num_decisions: int,
        capacity: float,
        penalty_remove: float,
        penalty_add: float,
        values_lb: float = 3.0,
        values_ub: float = 8.0,
        seed: int = 5,
        num_scenarios: int = 1,
        # dimension: int = 1,
    ):
        # Setting input parameters
        self.num_decisions = num_decisions
        self.capacity = capacity
        self.penalty_remove = penalty_remove
        self.penalty_add = penalty_add
        self.values_lb = values_lb
        self.values_ub = values_ub
        self.num_scenarios = num_scenarios
        self.seed = seed

        # Setting basic model parameters
        model_sense = "MAX"
        decision_variables = {
            "select_item": (num_decisions,),
        }
        #'y_add': (num_decisions, num_scenarios),
        #'y_remove': (num_decisions, num_scenarios)}

        _shape = (num_decisions, num_scenarios) if num_scenarios > 1 else (num_decisions,)
        param_to_predict_shapes = {"item_weights": _shape}
        extra_param_shapes = None

        # Setting additional model parameters
        np.random.seed(seed)
        self.capacity_np = np.array(capacity)
        self.values = np.random.uniform(values_lb, values_ub, num_decisions)

        GRBPYTwoStageModel.__init__(
            self,
            decision_variables,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=extra_param_shapes,
        )

    def _create_model(self):
        """Creates model AND variables_dict"""
        # Create a GP model
        self.gp_model = gp.Model("two_stage_knapsack")
        self.vars_dict = {}
        self.second_stage_vars_dict = {}

        # Define variables
        x = self.gp_model.addMVar(self.num_decisions, name="select_item", vtype=GRB.BINARY)
        y_add = self.gp_model.addMVar((self.num_decisions, self.num_scenarios), name="y_add", vtype=GRB.BINARY)
        y_remove = self.gp_model.addMVar((self.num_decisions, self.num_scenarios), name="y_remove", vtype=GRB.BINARY)
        self.vars_dict["select_item"] = x
        self.second_stage_vars_dict["y_remove"] = y_remove
        self.second_stage_vars_dict["y_add"] = y_add

        # It is a maximization problem
        self.gp_model.modelSense = self.model_sense_int

        # Set constraints
        self.gp_model.addConstrs(x[i] >= y_remove[i, j] for i in range(self.num_decisions) for j in range(self.num_scenarios))
        self.gp_model.addConstrs(x[i] <= 1 - y_add[i, j] for i in range(self.num_decisions) for j in range(self.num_scenarios))

        # Set objective
        obj = gp.quicksum(
            (self.values[i] * x[i] + self.values[i] * (y_add[i, j] * self.penalty_add - y_remove[i, j] * self.penalty_remove)) / self.num_scenarios
            for i in range(self.num_decisions)
            for j in range(self.num_scenarios)
        )

        self.gp_model.setObjective(obj)

    def _set_params(self, *parameters_i: np.ndarray):
        """
        Set parameters for two-stage knapsack, this corresponds to adjusting the weight scenarios in the constraints.
        """
        # Obtain the weight parameters
        weights = parameters_i[0]

        # Reshape the weights parameters
        weights = weights.reshape(-1, self.num_scenarios)

        # Max weight to overcome numerical issues
        max_weight = 10**6
        max_array = max_weight * np.ones(weights.shape)
        weights = np.minimum(weights, max_array)

        # Remove existing constraints
        self.gp_model.remove(self.gp_model.getConstrs())

        # Set new constraints
        x = self.vars_dict["select_item"]
        y_add = self.second_stage_vars_dict["y_add"]
        y_remove = self.second_stage_vars_dict["y_remove"]
        self.gp_model.addConstrs(
            gp.quicksum(weights[i, j] * (x[i] + y_add[i, j] - y_remove[i, j]) for i in range(self.num_decisions)) <= self.capacity_np
            for j in range(self.num_scenarios)
        )

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        # Since this function is to create a quadratic variant, we only care about first stage variables
        var_domain_dict = {"select_item": {"boolean": True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    @staticmethod
    def get_constraints(vars_dict: dict[str, cp.Variable]):
        # These constraints are cvxpy style constraints
        return []
