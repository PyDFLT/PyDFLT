
import numpy as np
import torch

from src.abstract_models.grbpy import GRBPYModel

from pyepo.model.grb.tsp import tspDFJModel



class GRBPYTravelingSalesperson(GRBPYModel, tspDFJModel):

    def __init__(self,
                 num_nodes: int,
                 num_scenarios: int = 1
                 ):
        """
        Creates a GRBPYTravelingSalesperson instance

        Args:
            num_nodes (int): Number of nodes
        """
        self.num_nodes = num_nodes
        self.num_scenarios = num_scenarios

        # Setting basic model parameters

        # Optimization model does not call its parents, so we do this to ensure
        # Pyepos tsp model innit gets called
        tspDFJModel.__init__(self, num_nodes=num_nodes)

        # triangle number for the number of edges
        num_edges = num_nodes * (num_nodes - 1) // 2
        model_sense = "MIN"
        var_shapes = {'x': (num_edges, )}
        param_to_predict_shapes = {'c': (num_edges, )}

        GRBPYModel.__init__(
            self,
            var_shapes,
            param_to_predict_shapes,
            model_sense,
        )

        assert num_edges == self.num_cost, (
             'The number of edges should equal the number of costs in the pyepo model\n' +
            f'But they where not: {num_edges} != {self.num_cost}'
        )

    def _set_params(self, params_i: np.ndarray):
        self.setObj(params_i)

    def get_objective(self,
                      data_batch: dict[str, torch.Tensor],
                      decisions_batch: dict[str, torch.Tensor],
                      predictions_batch: dict[str, torch.Tensor] = None
                      ) -> torch.float:
        return (data_batch['c'] * decisions_batch['x']).sum(-1)

    def _create_model(self):
        # Ensure we and pyepo use the same model
        gp_model = self._model
        vars_dict = {'x': [self.x[e] for e in self.edges]}

        return gp_model, vars_dict

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        var_domain_dict = {'x': {'boolean': True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    def _extract_decision_dict_i(self) -> dict[str, np.ndarray]:
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return {
            'x': sol
        }
