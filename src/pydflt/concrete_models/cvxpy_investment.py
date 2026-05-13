import cvxpy as cp
import torch

from pydflt.abstract_models.base import MAX
from pydflt.abstract_models.cvxpy_diff import CVXPYDiffModel


class CVXPYDiffInvestmentModel(CVXPYDiffModel):
    """
    A CVXPY-based differentiable portfolio investment optimization model.
    This model solves a portfolio allocation problem where the goal is to maximize
    the expected log-return of a portfolio subject to investment constraints.

    Attributes:
        num_decisions (int): Number of assets (decision variables) in the portfolio.
        bank_return (float): Risk-free return rate of the bank account (uninvested capital).
        num_scenarios (int): Number of scenarios for multi-scenario optimization.
    """

    def __init__(
        self,
        num_decisions: int,
        bank_return: float = 0.0,
        num_scenarios: int = 1,
    ):
        """
        Initializes the CVXPYDiffInvestmentModel.

        Args:
            num_decisions (int): Number of assets (decision variables) in the portfolio.
            bank_return (float): Risk-free return rate of the bank account. Defaults to 0.0.
            num_scenarios (int): Number of scenarios for multi-scenario optimization. Defaults to 1.
        """
        # Setting input parameters
        self.num_decisions = num_decisions
        self.bank_return = bank_return
        self.num_scenarios = num_scenarios

        # Setting basic model parameters
        model_sense = MAX
        var_shapes = {"investment": (num_decisions,)}
        _shape = (num_decisions, num_scenarios) if num_scenarios > 1 else (num_decisions,)
        param_to_predict_shapes = {"return": _shape}
        extra_param_shapes = None

        super().__init__(var_shapes, param_to_predict_shapes, model_sense, extra_param_shapes=extra_param_shapes)

    def _create_cp_model(self):
        """
        Creates the CVXPY optimization model for the portfolio investment problem.
        This method defines the decision variables, constraints, and objective function.

        Returns:
            cp.Problem: The CVXPY optimization problem instance.
        """
        x = self.cp_vars_dict["investment"]
        c = self.cp_params_dict["return"]
        constraints = [x >= 0, x <= 1, cp.sum(x) <= 1]
        obj = cp.sum(cp.log(1 + self.bank_return * (1 - cp.sum(x)) + x @ c))  # outside sum is for the scenarios

        return cp.Problem(cp.Maximize(obj), constraints)

    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.float:
        """
        Computes the objective function value for the portfolio investment problem.
        The objective is to maximize the expected log-return of the portfolio.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data, including 'c'.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing decision variables, including 'x'.
            predictions_batch (dict[str, torch.Tensor], optional): Unused for this implementation. Defaults to None.

        Returns:
            torch.float: The log-return of the portfolio for the batch.
        """
        c = data_batch["return"]
        x = decisions_batch["investment"]
        obj = torch.log(1 + (self.bank_return * (1 - x.sum(-1))) + (x * c).sum(-1))

        return obj

    def get_penalty(self, x_u: torch.Tensor, data_batch: dict[str, torch.Tensor]):
        """
        Computes the constraint violation penalty for the investment problem.
        Penalizes violations of the bounds x >= 0, x <= 1, and the budget constraint sum(x) <= 1.

        Args:
            x_u (torch.Tensor): Unconstrained decision variable tensor of shape (batch, num_decisions).
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data (unused here).

        Returns:
            torch.Tensor: The maximum constraint violation per sample, shape (batch,).
        """
        penalty_tensor = torch.concat(
            [
                torch.relu(0 - x_u),
                torch.relu(x_u - 1),
                torch.abs(x_u.sum(dim=1, keepdims=True) - 1),
            ],
            dim=1,
        )
        return penalty_tensor.amax(dim=1)

    def get_reward_gradient(self, decisions_batch: dict[str, torch.Tensor], data_batch: dict[str, torch.Tensor]):
        """
        Computes the gradient of the objective with respect to the decision variables.

        Args:
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing decision variables, including 'x'.
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data, including 'c'.

        Returns:
            torch.Tensor: The gradient of the objective w.r.t. x, shape (batch, num_decisions).
        """
        x = decisions_batch["investment"]
        c = data_batch["return"]
        # BUG: nom should be (c - self.bank_return) since d/dx_i of r*(1-sum(x)) = -r
        # BUG: denom should be the argument of the log, not log of it:
        #      1 + self.bank_return * (1 - x.sum(-1)) + (x * c).sum(-1), unsqueezed for broadcasting
        nom = self.bank_return * torch.ones(x.shape) + c
        denom = torch.log((1 + (self.bank_return * (1 - x.sum(-1))).unsqueeze(1) + x * c).sum(-1))
        return nom / denom
