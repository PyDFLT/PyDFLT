import unittest

import numpy as np

from src.concrete_models.grbpy_knapsack import GRBPYKnapsackModel


class TestQuadratic(unittest.TestCase):
    def test_quadratic_proxy(self):
        """
        Tests the creation and behavior of the quadratic proxy model.
        The quadratic proxy should minimize the L2 distance to a target vector,
        subject to the original model's constraints.
        """
        # 1. Arrange: Set up the original model and a target vector
        num_decisions = 5
        capacity = 7
        # Use constant weights for simplicity. The constraint will be: 3 * sum(x) <= 7, so sum(x) <= 2
        opt_model = GRBPYKnapsackModel(
            num_decisions=num_decisions,
            capacity=capacity,
            weights_lb=3.0,
            weights_ub=3.0,
            seed=42,  # for reproducibility of weights
        )

        # The target vector `w` has three items we want to select.
        # However, the capacity constraint only allows for two items.
        target_vector = np.array([1, 1, 1, 0, 0])

        # 2. Act: Create the quadratic proxy and solve it with the target vector
        qp_model = opt_model.create_quadratic_variant()
        decisions_dict = qp_model._solve_sample(target_vector)
        decision_x = decisions_dict["select_item"]

        # 3. Assert: The decision should be the closest feasible point to the target
        # The model must select 2 items to satisfy the capacity constraint (sum(x) <= 2).
        self.assertEqual(sum(decision_x), 2)

        # To be closest to the target [1, 1, 1, 0, 0], the decision must select two of the first three items.
        # This means the last two items must not be selected, as that would increase the distance.
        self.assertEqual(decision_x[3], 0)
        self.assertEqual(decision_x[4], 0)

        # The sum of the first three items in the decision must be 2.
        self.assertEqual(sum(decision_x[:3]), 2)


if __name__ == "__main__":
    unittest.main()
