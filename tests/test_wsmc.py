import unittest

import numpy as np

from src.concrete_models.grbpy_two_stage_weighted_set_multi_cover import (
    WeightedSetMultiCover,
)


class TestWSMC(unittest.TestCase):
    def test_construction_model(self):
        # Define opt WSMC model object
        opt_model = WeightedSetMultiCover(
            num_items=2,
            num_covers=3,
            penalty=5,
            cover_costs_lb=5,
            cover_costs_ub=50,
            seed=5,
            num_scenarios=1,
            recovery_ratio=0,
        )

        # Create a similar object with the same seed
        opt_model_same_seed = WeightedSetMultiCover(
            num_items=2,
            num_covers=3,
            penalty=5,
            cover_costs_lb=5,
            cover_costs_ub=50,
            seed=5,
            num_scenarios=1,
            recovery_ratio=0,
        )

        for i in range(opt_model.num_scenarios):
            self.assertEqual(opt_model.cover_costs[i], opt_model_same_seed.cover_costs[i])

        print(opt_model.item_cover_matrix)

    def test_use_opt_model_first_stage(self):
        # Test it selects the only cover that has the right item
        opt_model = WeightedSetMultiCover(
            num_items=2,
            num_covers=3,
            penalty=100,
            cover_costs_lb=5,
            cover_costs_ub=50,
            seed=5,
            num_scenarios=1,
            recovery_ratio=0,
        )
        opt_model.cover_costs = np.array([2, 2, 100])
        opt_model.cover_costs = np.array([[1, 0], [0, 1], [0, 0]])
        decisions_dict = opt_model._solve_sample(np.array([1, 0]))
        self.assertEqual(decisions_dict["select_cover"][0], 1)

        # Test it selects the only cover that has the right item
        opt_model = WeightedSetMultiCover(
            num_items=2,
            num_covers=3,
            penalty=100,
            cover_costs_lb=5,
            cover_costs_ub=50,
            seed=5,
            num_scenarios=1,
            recovery_ratio=0,
        )
        opt_model.cover_costs = np.array([2, 2, 100])
        opt_model.cover_costs = np.array([[1, 0], [0, 1], [0, 0]])
        decisions_dict = opt_model._solve_sample(np.array([0, 1]))
        self.assertEqual(decisions_dict["select_cover"][1], 1)


if __name__ == "__main__":
    unittest.main()
