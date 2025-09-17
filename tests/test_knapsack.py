import unittest

import numpy as np

from src.concrete_models.grbpy_knapsack import GRBPYKnapsackModel
from src.generate_data_functions.generate_data_knapsack import gen_data_knapsack


class TestKnapsack(unittest.TestCase):
    def test_construction_model(self):
        # Define opt knapsack model
        num_decisions = 10
        capacity = 20
        opt_model = GRBPYKnapsackModel(
            num_decisions=num_decisions,
            capacity=capacity,
            weights_lb=3.0,
            weights_ub=8.0,
            seed=5,
        )
        self.assertEqual(opt_model.num_decisions, num_decisions)
        self.assertEqual(opt_model.capacity, capacity)

    def test_use_opt_model_no_items(self):
        # Test select no item
        num_decisions = 3
        capacity = 2
        opt_model = GRBPYKnapsackModel(
            num_decisions=num_decisions,
            capacity=capacity,
            weights_lb=3.0,
            weights_ub=3.0,
            seed=5,
        )
        decisions_dict = opt_model._solve_sample(np.array([10, 5, 4]))
        self.assertEqual(sum(decisions_dict["select_item"]), 0)

    def test_use_opt_model_all_items(self):
        # Test select no item
        num_decisions = 3
        capacity = 9
        opt_model = GRBPYKnapsackModel(
            num_decisions=num_decisions,
            capacity=capacity,
            weights_lb=3.0,
            weights_ub=3.0,
            seed=5,
        )
        decisions_dict = opt_model._solve_sample(np.array([10, 5, 4]))
        self.assertEqual(sum(decisions_dict["select_item"]), 3)

    def test_use_opt_one_item(self):
        # Test select no item
        num_decisions = 3
        capacity = 3
        opt_model = GRBPYKnapsackModel(
            num_decisions=num_decisions,
            capacity=capacity,
            weights_lb=3.0,
            weights_ub=3.0,
            seed=5,
        )
        decisions_dict = opt_model._solve_sample(np.array([10, 5, 4]))

        # Test that only item is selected
        self.assertEqual(sum(decisions_dict["select_item"]), 1)

        # Test that the item with the highest value is selected
        self.assertEqual(decisions_dict["select_item"][0], 1)

    def test_generate_data_seed(self):
        """
        Test if setting the seed results in the same data
        """
        data_knapsack_one = gen_data_knapsack(seed=1, num_data=3, num_features=1, num_items=1)
        data_knapsack_two = gen_data_knapsack(seed=1, num_data=3, num_features=1, num_items=1)
        self.assertEqual(data_knapsack_one["features"][0], data_knapsack_two["features"][0])
        self.assertEqual(data_knapsack_one["item_value"][0], data_knapsack_two["item_value"][0])
        self.assertEqual(data_knapsack_one["features"][1], data_knapsack_two["features"][1])
        self.assertEqual(data_knapsack_one["item_value"][1], data_knapsack_two["item_value"][1])


if __name__ == "__main__":
    unittest.main()
