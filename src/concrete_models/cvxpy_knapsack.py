import cvxpy as cp
import numpy as np
import torch

from src.abstract_models.cvxpy_diff import CVXPYDiffModel


class CVXPYDiffKnapsackModel(CVXPYDiffModel):
    """
    A CVXPY-based differentiable knapsack optimization model.
    This model solves a knapsack problem where the goal is to maximize the total value
    of selected items subject to capacity constraints, allowing partial item selection.

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
        Initializes the CVXPYDiffKnapsackModel.

        Args:
            num_decisions (int): Number of items (decision variables) in the knapsack.
            capacity (float): The capacity constraint of the knapsack.
            weights_lb (float): Lower bound for item weights used in random weight generation. Defaults to 3.0.
            weights_ub (float): Upper bound for item weights used in random weight generation. Defaults to 8.0.
            dimension (int): Dimension of the weights for the items. Defaults to 1.
            seed (int): Random seed for reproducible weight generation. Defaults to 5.
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
        var_shapes = {"select_item": (num_decisions,)}
        _shape = (num_decisions, num_scenarios) if num_scenarios > 1 else (num_decisions,)
        param_to_predict_shapes = {"item_value": _shape}
        extra_param_shapes = None

        # Setting additional model parameters
        np.random.seed(seed)
        self.weights = np.random.uniform(weights_lb, weights_ub, (dimension, num_decisions))
        self.capacity_np = self.capacity * np.ones(dimension)

        super().__init__(
            var_shapes,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=extra_param_shapes,
        )

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
        v = data_batch["item_value"]
        x = decisions_batch["select_item"]
        return (v * x).sum(-1)

    def _create_cp_model(self):
        """
        Creates the CVXPY optimization model for the knapsack problem.
        This method defines the decision variables, constraints, and objective function.

        Returns:
            cp.Problem: The CVXPY optimization problem instance.
        """
        x_var = self.cp_vars_dict["select_item"]
        v_par = self.cp_params_dict["item_value"]
        constraints = [x_var >= 0, x_var <= 1, self.weights @ x_var <= self.capacity]
        obj = cp.sum(v_par @ x_var)

        return cp.Problem(cp.Maximize(obj), constraints)
