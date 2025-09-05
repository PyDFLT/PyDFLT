import cvxpy as cp
import numpy as np
import torch

from src.abstract_models.cvxpy_diff import CVXPYDiffModel


class CVXPYDiffKnapsackModel(CVXPYDiffModel):
    def __init__(
        self,
        num_decisions: int,
        capacity: float,
        values_lb: float = 3.0,
        values_ub: float = 8.0,
        dimension: int = 1,
        seed: int = 5,
        num_scenarios: int = 1,
    ):
        # Setting input parameters
        self.num_decisions = num_decisions
        self.capacity = capacity
        self.values_lb = values_lb
        self.values_lb = values_ub
        self.dimension = dimension
        self.seed = seed
        self.num_scenarios = num_scenarios

        # Setting basic model parameters
        model_sense = "MAX"
        var_shapes = {"x": (num_decisions,)}
        _shape = (num_decisions, num_scenarios) if num_scenarios > 1 else (num_decisions,)
        param_to_predict_shapes = {"c": _shape}
        extra_param_shapes = None

        # Setting additional model parameters
        np.random.seed(seed)
        self.weights = np.random.uniform(values_lb, values_ub, (dimension, num_decisions))
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
        c = data_batch["c"]
        x = decisions_batch["x"]
        return (c * x).sum(-1)

    def _create_cp_model(self):
        x_var = self.cp_vars_dict["x"]
        c_par = self.cp_params_dict["c"]
        constraints = [x_var >= 0, x_var <= 1, self.weights @ x_var <= self.capacity]
        obj = cp.sum(c_par @ x_var)
        self.cp_model = cp.Problem(cp.Maximize(obj), constraints)
