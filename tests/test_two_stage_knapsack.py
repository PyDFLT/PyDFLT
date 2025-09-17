import unittest

import numpy as np

from src.concrete_models.grbpy_two_stage_knapsack import TwoStageKnapsack


class TestKnapsack(unittest.TestCase):
    def test_construction_model(self):
        # Define opt knapsack model
        num_decisions = 10
        capacity = 20
        opt_model = TwoStageKnapsack(
            num_decisions=num_decisions,
            capacity=capacity,
            values_lb=3.0,
            values_ub=8.0,
            seed=5,
            penalty_add=2,
            penalty_remove=2,
        )

        self.assertEqual(opt_model.num_decisions, num_decisions)
        self.assertEqual(opt_model.capacity, capacity)

    def test_use_opt_model_no_items(self):
        # Test select no item
        num_decisions = 3
        capacity = 3
        opt_model = TwoStageKnapsack(
            num_decisions=num_decisions,
            capacity=capacity,
            values_lb=3.0,
            values_ub=8.0,
            seed=5,
            penalty_add=2,
            penalty_remove=2,
        )
        decisions_dict = opt_model._solve_sample(np.array([10, 5, 4]))
        self.assertEqual(sum(decisions_dict["select_item"]), 0)

    def test_use_opt_model_all_items(self):
        # Test select no item
        num_decisions = 3
        capacity = 9
        opt_model = TwoStageKnapsack(
            num_decisions=num_decisions,
            capacity=capacity,
            values_lb=3.0,
            values_ub=3.0,
            seed=5,
            penalty_add=0,
            penalty_remove=0,
        )

        predicted_weights = np.array([3, 3, 3])
        actual_weights = [predicted_weights]
        decisions_dict = opt_model._solve_sample(predicted_weights)

        opt_model.solve_second_stage(decisions_dict, *actual_weights)
        self.assertEqual(sum(decisions_dict["select_item"]), 3)

    def test_use_opt_model_no_items_and_zero_obj(self):
        # Test select no item
        num_decisions = 3
        capacity = 1
        opt_model = TwoStageKnapsack(
            num_decisions=num_decisions,
            capacity=capacity,
            values_lb=3.0,
            values_ub=3.0,
            seed=5,
            penalty_add=0,
            penalty_remove=10,
        )

        predicted_weights = np.array([3, 3, 3])
        actual_weights = [predicted_weights]
        decisions_dict = opt_model._solve_sample(predicted_weights)

        obj = opt_model.solve_second_stage(decisions_dict, *actual_weights)
        self.assertEqual(sum(decisions_dict["select_item"]), 0)
        self.assertEqual(obj["objective_value"], 0)


if __name__ == "__main__":
    unittest.main()
