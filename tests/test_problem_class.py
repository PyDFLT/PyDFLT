import unittest

import torch

from src.concrete_models.grbpy_shortest_path import ShortestPath
from src.concrete_models.grbpy_two_stage_weighted_set_multi_cover import (
    WeightedSetMultiCover,
)
from src.decision_makers import DifferentiableDecisionMaker
from src.generate_data_functions.generate_data_shortest_path import gen_data_shortest_path
from src.generate_data_functions.generate_data_wsmc import gen_data_wsmc
from src.problem import Problem
from src.runner import Runner


class TestProblemClass(unittest.TestCase):
    def test_problem_class(self):
        """
        This tests checks whether the test set is the same when we use the same seed
        """
        seed = 1
        num_covers = 2
        num_items = 2

        # Create a dummy model
        opt_model = WeightedSetMultiCover(
            num_items=num_items,
            num_covers=num_covers,
            seed=seed,
            penalty=1,
            cover_costs_lb=1,
            cover_costs_ub=3,
            recovery_ratio=0,
        )
        data_dict = gen_data_wsmc(
            seed=1,
            num_data=10,
            num_features=1,
            num_items=num_items,
            degree=5,
            noise_width=0.5,
        )
        problem = Problem(data_dict=data_dict, opt_model=opt_model, train_ratio=0.5, val_ratio=0.4)
        problem.set_mode("test")
        indices_test = problem.generate_batch_indices(batch_size=1)
        data = problem.read_data(indices_test)

        # Create the same model again
        opt_model = WeightedSetMultiCover(
            num_items=num_items,
            num_covers=num_covers,
            seed=seed,
            penalty=1,
            cover_costs_lb=1,
            cover_costs_ub=3,
            recovery_ratio=0,
        )
        data_dict = gen_data_wsmc(
            seed=1,
            num_data=10,
            num_features=1,
            num_items=num_items,
            degree=5,
            noise_width=0.5,
        )
        problem = Problem(data_dict=data_dict, opt_model=opt_model, train_ratio=0.5, val_ratio=0.4)
        problem.set_mode("test")
        indices_test_2 = problem.generate_batch_indices(batch_size=1)
        data_2 = problem.read_data(indices_test)

        # Check indices are the same
        self.assertEqual(indices_test[0], indices_test_2[0])

        # Check data is the same
        coverage_equal = torch.equal(data["coverage_requirements"][0][0], data_2["coverage_requirements"][0][0])
        features_equal = torch.equal(data["features"][0][0], data_2["features"][0][0])
        self.assertTrue(coverage_equal)
        self.assertTrue(features_equal)

    def test_robust_losses(self):
        """
        This tests checks if the robust losses work correctly
        """

        # Create a dummy model
        opt_model = ShortestPath(grid=(5, 5))
        data_dict = gen_data_shortest_path(num_data=20, num_features=1, grid=(5, 5))
        problem = Problem(
            data_dict=data_dict,
            opt_model=opt_model,
            train_ratio=0.5,
            val_ratio=0.4,
            knn_robust_loss=5,
            knn_robust_loss_weight=0.5,
        )
        decision_maker = DifferentiableDecisionMaker(
            problem=problem,
            learning_rate=0.001,
            batch_size=32,
            device_str="cpu",
            loss_function_str="SPOPlus",
            predictor_str="MLP",
        )
        runner = Runner(decision_maker, num_epochs=2, use_wandb=False)
        runner.run()


if __name__ == "__main__":
    unittest.main()
