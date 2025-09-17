import unittest

from src.concrete_models.grbpy_shortest_path import ShortestPath
from src.generate_data_functions.generate_data_shortestpath import gen_data_shortestpath


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
        data_sp_one = gen_data_shortestpath(seed=1, num_data=2, grid=(2, 2), num_features=1)
        print(data_sp_one)
        data_sp_two = gen_data_shortestpath(seed=1, num_data=2, grid=(2, 2), num_features=1)

        self.assertEqual(data_sp_one["features"][0], data_sp_two["features"][0])
        self.assertEqual(data_sp_one["arc_costs"][0][0], data_sp_two["arc_costs"][0][0])
        self.assertEqual(data_sp_one["features"][1], data_sp_two["features"][1])
        self.assertEqual(data_sp_one["arc_costs"][1][0], data_sp_two["arc_costs"][1][0])


if __name__ == "__main__":
    unittest.main()
