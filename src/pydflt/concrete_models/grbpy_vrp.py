import numpy as np
import torch

from pydflt.abstract_models.grbpy import GRBPYModel
from pydflt.utils.vrp import VrpModel


class VehicleRouting(GRBPYModel, VrpModel):
    """
    Gurobi-based capacitated Vehicle Routing Problem (VRP) using the PyEPO vrpModel.

    The model minimizes total edge cost subject to vehicle capacity and routing constraints.
    """

    def __init__(
        self,
        num_nodes: int,
        capacity: int,
        num_vehicles: int,
        demands_lb: float = 0.0,
        demands_ub: float = 10.0,
        seed: int = 5,
        num_scenarios: int = 1,
        scale_demand: float = 0.7,
        verbose: bool = False,
        time_limit: float | None = None,
    ):
        """
        Args:
            num_nodes: Number of nodes (including depot).
            capacity: Vehicle capacity.
            num_vehicles: Number of vehicles.
            demands_lb: Lower bound for random demand generation.
            demands_ub: Upper bound for random demand generation.
            seed: RNG seed for demand generation.
            num_scenarios: Number of scenarios for multi-scenario optimization (unused, kept for API symmetry).
        """
        self.num_nodes = num_nodes
        self.capacity = capacity
        self.num_vehicles = num_vehicles
        self.demands_lb = demands_lb
        self.demands_ub = demands_ub
        self.num_scenarios = num_scenarios
        self.scale_demand = scale_demand
        self.verbose = verbose
        self.time_limit = time_limit

        # Setting additional model parameters
        rng = np.random.default_rng(seed)
        self.demands = np.round(self.demands_lb + rng.random(self.num_nodes - 1) * (self.demands_ub - self.demands_lb))
        if scale_demand > 0:
            total_demand = sum(self.demands)
            total_goal_demand = capacity * num_vehicles * scale_demand
            self.demands = self.demands * total_goal_demand / total_demand
            assert max(self.demands) <= capacity, "Vrp parameters cause a single demand to exceed capacity"

        # Initialize the underlying PyEPO VRP model
        VrpModel.__init__(self, num_nodes=num_nodes, demands=self.demands, capacity=capacity, num_vehicles=num_vehicles)

        num_edges = len(self.edges)
        model_sense = "MIN"
        var_shapes = {"select_edge": (num_edges,)}
        param_to_predict_shapes = {"edge_costs": (num_edges,)}

        GRBPYModel.__init__(
            self,
            var_shapes,
            param_to_predict_shapes,
            model_sense,
            verbose=verbose,
            time_limit=time_limit,
        )
        self.supports_binding_constraints = True
        self.supports_adjacent_vertices = True

        assert num_edges == self.num_cost, (
            "The number of edges should equal the number of costs in the pyepo model\n" + f"But they were not: {num_edges} != {self.num_cost}"
        )

        # Use the VRP lazy constraint callback
        self.lazy_constraints_method = self._vrp_callback

    def _set_params(self, params_i: np.ndarray) -> None:
        """
        Set edge costs for a single instance.
        """
        # Reset lazy constraint bookkeeping for this solve:
        self.setObj(params_i)

    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Total cost of selected edges.
        """
        return (data_batch["edge_costs"] * decisions_batch["select_edge"]).sum(-1)

    def _create_model(self):
        """
        Reuse the PyEPO VRP model to ensure consistent formulation.
        """
        gp_model = self._model
        vars_dict = {"select_edge": [self.x[e] for e in self.edges]}
        return gp_model, vars_dict

    def _set_optmodel_attributes(self) -> None:
        self.modelSense = self.model_sense_int
        self._model = self.gp_model

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        return {"select_edge": {"boolean": True}}

    def _extract_decision_dict_i(self) -> dict[str, np.ndarray]:
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:
                sol[i] = 1
        return {"select_edge": sol}
