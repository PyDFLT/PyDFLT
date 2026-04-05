import unittest

import numpy as np
from pydflt.concrete_models.grbpy_vrp import VehicleRouting


class TestGRBPYVRP(unittest.TestCase):
    def test_two_clusters_two_vehicles(self):
        # 1 depot + 4 customers
        num_nodes = 5
        num_vehicles = 2
        capacity = 10
        model = VehicleRouting(
            num_nodes=num_nodes,
            capacity=capacity,
            num_vehicles=num_vehicles,
            seed=1,
            scale_demand=0.0,
            time_limit=5.0,
        )

        # Overwrite demands: customers 1..4
        model.demands = np.array([6.0, 2.0, 5.0, 3.0])
        model._model._q = {i + 1: float(d) for i, d in enumerate(model.demands)}

        # Coordinates: two tight pairs far apart
        coords = {
            0: (0.0, 0.0),
            1: (1.0, 0.0),
            2: (1.0, 1.0),
            3: (10.0, 0.0),
            4: (10.0, 1.0),
        }

        edge_costs = np.zeros(len(model.edges), dtype=np.float32)
        for idx, (i, j) in enumerate(model.edges):
            xi, yi = coords[i]
            xj, yj = coords[j]
            edge_costs[idx] = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5

        decision_dict = model._solve_sample(edge_costs)
        selected_edges = decision_dict["select_edge"]

        routes = model.getTour(selected_edges)
        assert len(routes) == 2

        customer_sets = [set(route[1:-1]) for route in routes]
        expected = [{1, 2}, {3, 4}]

        assert set(map(frozenset, customer_sets)) == set(map(frozenset, expected))


if __name__ == "__main__":
    unittest.main()
