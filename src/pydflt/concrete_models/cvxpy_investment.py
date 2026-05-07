from src.pydflt.abstract_models.cvxpy_diff import CVXPYDiffModel
from src.pydflt.abstract_models.base import MAX
import cvxpy as cp
import torch


class CVXPYDiffInvestmentModel(CVXPYDiffModel):
    def __init__(self,
                 num_decisions: int,
                 bank_return: float = 0.0,
                 num_scenarios: int = 1
                 ):
        # Setting input parameters
        self.num_decisions = num_decisions
        self.bank_return = bank_return
        self.num_scenarios = num_scenarios

        # Setting basic model parameters
        model_sense = MAX
        var_shapes = {'x': (num_decisions, )}
        _shape = (num_decisions, num_scenarios) if num_scenarios > 1 else (num_decisions, )
        param_to_predict_shapes = {'c': _shape}
        extra_param_shapes = None

        super().__init__(var_shapes, param_to_predict_shapes, model_sense, extra_param_shapes=extra_param_shapes)

    def _create_cp_model(self):
        x = self.cp_vars_dict['x']
        c = self.cp_params_dict['c']
        constraints = [x >= 0, x <= 1, cp.sum(x) <= 1]
        obj = cp.sum(cp.log(1 + self.bank_return*(1-cp.sum(x)) + x @ c))  # outside sum is for the scenarios
        # obj = cp.sum(x @ c)
        self.cp_model = cp.Problem(cp.Maximize(obj), constraints)

    def get_objective(self,
                      data_batch: dict[str, torch.Tensor],
                      decisions_batch: dict[str, torch.Tensor],
                      predictions_batch: dict[str, torch.Tensor] = None
                      ) -> torch.float:
        c = data_batch['c']
        x = decisions_batch['x']
        obj = torch.log((1+(self.bank_return * (1-x.sum(-1))) + (x*c).sum(-1)))
        # obj = (x * c).sum(-1)
        return obj

    def get_penalty(self, x_u: torch.Tensor, data_batch: dict[str, torch.Tensor]):
        penalty_tensor = torch.concat([torch.relu(0 - x_u), torch.relu(x_u - 1),
                                       torch.abs(x_u.sum(dim=1, keepdims=True) - 1)], dim=1)
        return penalty_tensor.amax(dim=1)

    def get_reward_gradient(self, decisions_batch: dict[str, torch.Tensor], data_batch: dict[str, torch.Tensor]):
        x = decisions_batch['x']
        c = data_batch['c']
        nom = self.bank_return * torch.ones(x.shape) + c
        # print('nom shape', nom.shape)
        denom = torch.log((1 + (self.bank_return * (1-x.sum(-1))).unsqueeze(1) + x*c).sum(-1))
        # print('denom shape', denom.shape)
        return nom / denom
