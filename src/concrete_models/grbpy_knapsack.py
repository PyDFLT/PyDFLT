import cvxpy as cp
import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel

from src.abstract_models.grbpy import GRBPYModel


class GRBPYKnapsackModel(GRBPYModel, optGrbModel):
    """
    A Gurobi-based knapsack optimization model.
    This model solves a knapsack problem where the goal is to maximize the total value
    of selected items subject to capacity constraints using the Gurobi solver.

    Attributes:
        num_decisions (int): Number of items (decision variables) in the knapsack.
        capacity (float): The capacity constraint of the knapsack.
        weights_lb (float): Lower bound for item weights during random generation.
        weights_ub (float): Upper bound for item weights during random generation.
        dimension (int): Dimension of the weights for the items.
        seed (int): Random seed for reproducible weight generation.
        num_scenarios (int): Number of scenarios for multi-scenario optimization.
        weights (np.ndarray): Fixed weights for the knapsack items.
        capacity_np (np.ndarray): Capacity constraints as a numpy array.
    """

    def __init__(
        self,
        num_decisions: int,
        capacity: float,
        weights_lb: float = 3.0,
        weights_ub: float = 8.0,
        dimension: int = 1,
        seed: int = 5,
        num_scenarios: int = 1,
    ):
        """
        Initializes the GRBPYKnapsackModel.

        Args:
            num_decisions (int): Number of items (decision variables) in the knapsack.
            capacity (float): The capacity constraint of the knapsack.
            weights_lb (float): Lower bound for item weights during random generation. Defaults to 3.0.
            weights_ub (float): Upper bound for item weights during random generation. Defaults to 8.0.
            dimension (int): Dimension of the weights for the items. Defaults to 1.
            seed (int): Random seed for reproducible weight generation. Defaults to None.
            num_scenarios (int): Number of scenarios for multi-scenario optimization. Defaults to 1.
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
        """
        Creates the Gurobi optimization model for the knapsack problem.
        This method defines the decision variables, constraints, and model sense.

        Returns:
            tuple: A tuple containing the Gurobi model and the variables dictionary.
        """
        # Create a GP model
        gp_model = gp.Model("knapsack")
        vars_dict = {}

        # Define vars
        name = "select_item"
        x = gp_model.addMVar(self.num_vars, name=name, vtype=GRB.BINARY)
        vars_dict[name] = x
        # It is a maximization problem
        gp_model.modelSense = self.modelSense = self.model_sense_int

        # Constraints
        gp_model.addConstr(self.weights @ x <= self.capacity_np)

        return gp_model, vars_dict

    def _set_params(self, *params_i: np.ndarray):
        """
        Sets the parameters for the knapsack model for a single instance.
        Updates the objective function with the provided item values.

        Args:
            *params_i (np.ndarray): Item values for the current instance.
        """
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
        """
        Computes the objective function value for the knapsack problem.
        The objective is to maximize the total value of selected items.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data, including 'item_value'.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing decision variables, including 'select_item'.
            predictions_batch (dict[str, torch.Tensor], optional): Unused for this implementation. Defaults to None.

        Returns:
            torch.float: The total value of selected items for the batch.
        """
        return (data_batch["item_value"] * decisions_batch["select_item"]).sum(-1)

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
        return self.gp_model, self.vars_dict["select_item"]

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        """
        Returns the variable domains for the knapsack problem when creating a quadratic variant.
        Specifies that the decision variables are boolean.

        Returns:
            dict[str, dict[str, bool]]: A dictionary specifying variable domains.
        """
        # Since this function is to create a quadratic variant, we only care about first stage variables
        var_domain_dict = {"select_item": {"boolean": True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    def get_constraints(self, vars_dict: dict[str, cp.Variable]):
        """
        Returns the constraints for the knapsack problem in CVXPY format.
        Used when creating a quadratic variant.

        Args:
            vars_dict (dict[str, cp.Variable]): A dictionary mapping variable names to CVXPY variables.

        Returns:
            list: A list of CVXPY constraints for the knapsack problem.
        """
        # These constraints are cvxpy style constraints
        x = vars_dict["select_item"]
        return [self.weights @ x <= self.capacity_np]
