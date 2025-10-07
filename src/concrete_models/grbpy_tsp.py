import numpy as np
import torch
from pyepo.model.grb.tsp import tspDFJModel

from src.abstract_models.grbpy import GRBPYModel


class TravelingSalesperson(GRBPYModel, tspDFJModel):
    """
    A Gurobi-based Traveling Salesperson Problem (TSP) optimization model.
    This model solves the TSP using the Dantzig-Fulkerson-Johnson (DFJ) formulation
    to find the shortest Hamiltonian cycle visiting all nodes exactly once.

    Attributes:
        num_nodes (int): Number of nodes in the TSP instance.
        num_scenarios (int): Number of scenarios for multi-scenario optimization.
    """

    def __init__(self, num_nodes: int, num_scenarios: int = 1):
        """
        Initializes the TravelingSalesperson model.

        Args:
            num_nodes (int): Number of nodes in the TSP instance.
            num_scenarios (int): Number of scenarios for multi-scenario optimization. Defaults to 1.
        """
        self.num_nodes = num_nodes
        self.num_scenarios = num_scenarios

        # Setting basic model parameters

        # Optimization model does not call its parents, so we do this to ensure PyEPO's tsp model innit gets called
        tspDFJModel.__init__(self, num_nodes=num_nodes)

        # triangle number for the number of edges
        num_edges = num_nodes * (num_nodes - 1) // 2
        model_sense = "MIN"
        var_shapes = {"select_edge": (num_edges,)}
        param_to_predict_shapes = {"edge_costs": (num_edges,)}

        GRBPYModel.__init__(
            self,
            var_shapes,
            param_to_predict_shapes,
            model_sense,
        )

        assert num_edges == self.num_cost, (
            "The number of edges should equal the number of costs in the pyepo model\n" + f"But they where not: {num_edges} != {self.num_cost}"
        )

    def _set_params(self, params_i: np.ndarray):
        """
        Sets the parameters for the TSP model for a single instance.
        Updates the objective function with the provided edge costs.

        Args:
            params_i (np.ndarray): Edge costs for the current instance.
        """
        self.setObj(params_i)

    def get_objective(
        self, data_batch: dict[str, torch.Tensor], decisions_batch: dict[str, torch.Tensor], predictions_batch: dict[str, torch.Tensor] = None
    ) -> torch.float:
        """
        Computes the objective function value for the TSP problem.
        The objective is to minimize the total cost of selected edges in the tour.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data, including "edge_costs" .
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing decision variables, including 'select_edge'.
            predictions_batch (dict[str, torch.Tensor], optional): Unused for this implementation. Defaults to None.

        Returns:
            torch.float: The total cost of selected edges for the batch.
        """
        return (data_batch["edge_costs"] * decisions_batch["select_edge"]).sum(-1)

    def _create_model(self):
        """
        Creates the Gurobi optimization model for the TSP problem.
        This method reuses the PyEPO TSP model to ensure consistency.

        Returns:
            tuple: A tuple containing the Gurobi model and the variables dictionary.
        """
        # Ensure we and pyepo use the same model
        gp_model = self._model
        vars_dict = {"select_edge": [self.x[e] for e in self.edges]}

        return gp_model, vars_dict

    def _set_optmodel_attributes(self) -> None:
        self.modelSense = self.model_sense_int
        self._model = self.gp_model

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, bool]]:
        """
        Returns the domain specifications for the decision variables.

        Returns:
            dict[str, dict[str, bool]]: A dictionary specifying that 'select_edge' variables are boolean.
        """
        var_domain_dict = {"select_edge": {"boolean": True}}  # boolean, integer, nonneg, nonpos, pos, imag, complex
        return var_domain_dict

    def _extract_decision_dict_i(self) -> dict[str, np.ndarray]:
        """
        Extracts the decision variables from the solved optimization model.
        Converts the Gurobi solution to a binary array indicating which edges are selected.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the 'select_edge' binary array.
        """
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for i, e in enumerate(self.edges):
            if self.x[e].x > 1e-2:  # Use integer index i, not edge tuple e
                sol[i] = 1
        return {"select_edge": sol}
