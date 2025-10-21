import unittest

import numpy as np

from pydflt.concrete_models.grbpy_tsp import TravelingSalesperson


class TestTSP(unittest.TestCase):
    def test_construction_tsp_model(self):
        """Test that TSP model is constructed correctly."""
        num_nodes = 4
        tsp = TravelingSalesperson(num_nodes)

        # For 4 nodes, we should have 6 edges (triangle number: 4*3/2)
        expected_num_edges = num_nodes * (num_nodes - 1) // 2
        self.assertEqual(len(tsp.edges), expected_num_edges)
        self.assertEqual(tsp.num_nodes, num_nodes)

    def test_dummy_problem(self):
        """
        Test TSP with 4 points forming a square.

        Points are arranged as:
        0 --- 1
        |     |
        |     |
        3 --- 2

        The optimal tour should be 0->1->2->3->0 (or reverse) using edges:
        (0,1), (1,2), (2,3), (3,0) with cost 1 each
        All other edges ((0,2), (1,3)) have cost 10 (diagonals)
        """
        num_nodes = 4
        tsp = TravelingSalesperson(num_nodes)

        # Set all costs to high value initially
        edge_costs = np.full(tsp.param_to_predict_shapes["edge_costs"], 10.0)

        # Set costs for square perimeter edges to 1 (optimal path)
        # The edges in tsp.edges are ordered as: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        square_edges = [(0, 1), (1, 2), (2, 3), (0, 3)]  # edges forming the square

        for edge_idx, edge in enumerate(tsp.edges):
            if edge in square_edges or (edge[1], edge[0]) in square_edges:
                edge_costs[edge_idx] = 1.0

        # Solve the TSP
        decision_dict = tsp._solve_sample(edge_costs)
        selected_edges = decision_dict["select_edge"]

        # Check that we have a valid binary solution
        self.assertTrue(np.all((selected_edges == 0) | (selected_edges == 1)))

        # Check that exactly 4 edges are selected (forming a complete tour)
        self.assertEqual(np.sum(selected_edges), num_nodes)

        # Check that the total cost is 4 (optimal square perimeter)
        total_cost = np.sum(edge_costs * selected_edges)
        self.assertEqual(total_cost, 4.0)

        # Verify that only the square edges are selected
        selected_edge_tuples = []
        for edge_idx, selected in enumerate(selected_edges):
            if selected == 1:
                selected_edge_tuples.append(tsp.edges[edge_idx])

        # Convert to set for comparison (order doesn't matter)
        selected_set = set(selected_edge_tuples)
        expected_set = {(0, 1), (1, 2), (2, 3), (0, 3)}

        # Check that selected edges form the square (allowing for undirected edges)
        expected_undirected = set()
        for edge in expected_set:
            expected_undirected.add(edge)
            expected_undirected.add((edge[1], edge[0]))  # reverse direction

        selected_undirected = set()
        for edge in selected_set:
            selected_undirected.add(edge)
            selected_undirected.add((edge[1], edge[0]))

        # The intersection should contain all square edges
        self.assertTrue(expected_set.issubset(selected_undirected))


if __name__ == "__main__":
    unittest.main()
