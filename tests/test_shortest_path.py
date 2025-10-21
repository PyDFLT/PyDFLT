import unittest

import numpy as np

from pydflt.concrete_models.grbpy_shortest_path import ShortestPath
from pydflt.generate_data_functions import gen_data_shortest_path


class TestShortestPath(unittest.TestCase):
    def test_construction_sp_model(self):
        grid = (5, 5)
        sp = ShortestPath(grid)
        num_coef = grid[0] * (grid[1] - 1) + grid[1] * (grid[0] - 1)
        self.assertEqual(len(sp.arcs), num_coef)

    def test_generate_data_seed(self):
        """
        Test if setting the seed results in the same data
        """
        data_sp_one = gen_data_shortest_path(seed=1, num_data=2, num_features=1, grid=(2, 2))
        print(data_sp_one)
        data_sp_two = gen_data_shortest_path(seed=1, num_data=2, num_features=1, grid=(2, 2))

        self.assertEqual(data_sp_one["features"][0], data_sp_two["features"][0])
        self.assertEqual(data_sp_one["arc_costs"][0][0], data_sp_two["arc_costs"][0][0])
        self.assertEqual(data_sp_one["features"][1], data_sp_two["features"][1])
        self.assertEqual(data_sp_one["arc_costs"][1][0], data_sp_two["arc_costs"][1][0])

    def test_dummy_problem(self):
        # Create a simple 3x3 grid for easier testing
        sp = ShortestPath((3, 3))
        grid_rows, grid_cols = sp.grid

        # Set all costs to 1 initially
        arc_costs = np.ones(sp.param_to_predict_shapes["arc_costs"])

        # Set costs to 0 for the optimal path: top row (horizontal) + rightmost column (vertical)
        # The optimal path should be: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)

        # Find arcs in the top row (row 0, horizontal edges)
        for arc_idx, (from_node, to_node) in enumerate(sp.arcs):
            from_row = from_node // grid_cols
            from_col = from_node % grid_cols
            to_row = to_node // grid_cols
            to_col = to_node % grid_cols

            # Top row horizontal edges: (0,0)->(0,1), (0,1)->(0,2)
            if from_row == 0 and to_row == 0 and to_col == from_col + 1:
                arc_costs[arc_idx] = 0

            # Rightmost column vertical edges: (0,2)->(1,2), (1,2)->(2,2)
            if from_col == grid_cols - 1 and to_col == grid_cols - 1 and to_row == from_row + 1:
                arc_costs[arc_idx] = 0

        # Solve and check that we get the expected optimal solution
        decision_dict = sp._solve_sample(arc_costs)
        selected_arcs = decision_dict["select_arc"]

        # Check that we have a valid binary solution
        self.assertTrue(np.all((selected_arcs == 0) | (selected_arcs == 1)))

        # Check that exactly 4 arcs are selected (the optimal path length)
        expected_path_length = grid_rows + grid_cols - 2
        self.assertEqual(np.sum(selected_arcs), expected_path_length)

        # Check that all selected arcs have cost 0 (i.e., we found the optimal path)
        selected_cost = np.sum(arc_costs * selected_arcs)
        self.assertEqual(selected_cost, 0)


if __name__ == "__main__":
    unittest.main()
