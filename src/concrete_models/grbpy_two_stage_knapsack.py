import cvxpy as cp
import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.abstract_models.grbpy_two_stage import GRBPYTwoStageModel


class TwoStageKnapsack(GRBPYTwoStageModel):
    """
    A Gurobi-based two-stage knapsack optimization model.
    This model solves a two-stage knapsack problem where decisions are made in two stages:
    first stage selects items, and second stage allows adding or removing items with penalties.

    Attributes:
        num_decisions (int): Number of items (decision variables) in the knapsack.
        capacity (float): The capacity constraint of the knapsack.
        penalty_remove (float): Penalty for removing an item in the second stage.
        penalty_add (float): Penalty for adding an item in the second stage.
        values_lb (float): Lower bound for item values during random generation.
        values_ub (float): Upper bound for item values during random generation.
        num_scenarios (int): Number of scenarios for multi-scenario optimization.
        seed (int): Random seed for reproducible value generation.
        capacity_np (np.ndarray): Capacity constraint as a numpy array.
        values (np.ndarray): Fixed values for the knapsack items.
    """

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
        """
        Initializes the TwoStageKnapsack model.

        Args:
            num_decisions (int): Number of items (decision variables) in the knapsack.
            capacity (float): The capacity constraint of the knapsack.
            penalty_remove (float): Penalty for removing an item in the second stage.
            penalty_add (float): Penalty for adding an item in the second stage.
            values_lb (float): Lower bound for item values during random generation. Defaults to 3.0.
            values_ub (float): Upper bound for item values during random generation. Defaults to 8.0.
            seed (int): Random seed for reproducible value generation. Defaults to 5.
            num_scenarios (int): Number of scenarios for multi-scenario optimization. Defaults to 1.
        """
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
        """
        Creates the Gurobi optimization model for the two-stage knapsack problem.
        This method defines the first and second stage variables, constraints, and objective function.

        Returns:
            tuple: A tuple containing the Gurobi model and the variables dictionary.
        """
        # Create a GP model
        gp_model = gp.Model("two_stage_knapsack")
        vars_dict = {}
        second_stage_vars_dict = {}

        # Define variables
        x = gp_model.addMVar(self.num_decisions, name="select_item", vtype=GRB.BINARY)
        y_add = gp_model.addMVar((self.num_decisions, self.num_scenarios), name="y_add", vtype=GRB.BINARY)
        y_remove = gp_model.addMVar((self.num_decisions, self.num_scenarios), name="y_remove", vtype=GRB.BINARY)
        vars_dict["select_item"] = x
        second_stage_vars_dict["y_remove"] = y_remove
        second_stage_vars_dict["y_add"] = y_add

        # It is a maximization problem
        gp_model.modelSense = self.model_sense_int

        # Set constraints
        gp_model.addConstrs(x[i] >= y_remove[i, j] for i in range(self.num_decisions) for j in range(self.num_scenarios))
        gp_model.addConstrs(x[i] <= 1 - y_add[i, j] for i in range(self.num_decisions) for j in range(self.num_scenarios))

        # Set objective
        obj = gp.quicksum(
            (self.values[i] * x[i] + self.values[i] * (y_add[i, j] * self.penalty_add - y_remove[i, j] * self.penalty_remove)) / self.num_scenarios
            for i in range(self.num_decisions)
            for j in range(self.num_scenarios)
        )

        gp_model.setObjective(obj)

        self.second_stage_vars_dict = second_stage_vars_dict

        return gp_model, vars_dict

    def _set_params(self, *parameters_i: np.ndarray):
        """
        Sets the parameters for the two-stage knapsack model for a single instance.
        This corresponds to adjusting the weight scenarios in the constraints.

        Args:
            *parameters_i (np.ndarray): Item weights for the current instance scenarios.
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
        """
        Returns the variable domains for the two-stage knapsack problem when creating a quadratic variant.
        Only considers first stage variables since that's what's relevant for the quadratic variant.

        Returns:
            dict[str, dict[str, bool]]: A dictionary specifying variable domains for first stage variables.
        """
        # Since this function is to create a quadratic variant, we only care about first stage variables
        var_domain_dict = {"select_item": {"boolean": True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    @staticmethod
    def get_constraints(vars_dict: dict[str, cp.Variable]):
        """
        Returns the constraints for the two-stage knapsack problem in CVXPY format.
        Used when creating a quadratic variant.

        Args:
            vars_dict (dict[str, cp.Variable]): A dictionary mapping variable names to CVXPY variables.

        Returns:
            list: An empty list since no additional constraints are needed for the quadratic variant.
        """
        # These constraints are cvxpy style constraints
        return []
