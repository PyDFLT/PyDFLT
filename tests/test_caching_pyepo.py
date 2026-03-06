import unittest

from pydflt.concrete_models.grbpy_shortest_path import ShortestPath
from pydflt.decision_makers import DifferentiableDecisionMaker
from pydflt.generate_data_functions import gen_data_shortest_path
from pydflt.problem import Problem
from pydflt.runner import Runner


class TestPyEPOCaching(unittest.TestCase):
    def test_spoplus_uses_solution_pool_cache(self):
        opt_model = ShortestPath(grid=(4, 4))
        data_dict = gen_data_shortest_path(num_data=12, num_features=2, grid=(4, 4), seed=1)
        problem = Problem(
            data_dict=data_dict,
            opt_model=opt_model,
            train_ratio=0.6,
            val_ratio=0.2,
            compute_optimal_decisions=True,
            compute_optimal_objectives=True,
            pyepo_solve_ratio=0.5,
        )
        decision_maker = DifferentiableDecisionMaker(
            problem=problem,
            learning_rate=1e-3,
            batch_size=4,
            device_str="cpu",
            predictor_str="MLP",
            predictor_kwargs={"num_hidden_layers": 0},
            loss_function_str="SPOPlus",
        )

        self.assertAlmostEqual(decision_maker.loss_function.solve_ratio, 0.5)
        self.assertIsNotNone(problem.solution_pool)
        self.assertIs(decision_maker.loss_function.solpool, problem.solution_pool)

        runner = Runner(decision_maker, num_epochs=1, use_wandb=False)
        result = runner.run()
        self.assertIsNotNone(result)

    def test_spoplus_cache_at_val_zero_solve_ratio(self):
        opt_model = ShortestPath(grid=(4, 4))
        data_dict = gen_data_shortest_path(num_data=10, num_features=2, grid=(4, 4), seed=2)
        problem = Problem(
            data_dict=data_dict,
            opt_model=opt_model,
            train_ratio=0.6,
            val_ratio=0.2,
            compute_optimal_decisions=True,
            compute_optimal_objectives=True,
            pyepo_solve_ratio=0.0,
            cache_at_val=True,
        )
        decision_maker = DifferentiableDecisionMaker(
            problem=problem,
            learning_rate=1e-3,
            batch_size=4,
            device_str="cpu",
            predictor_str="MLP",
            predictor_kwargs={"num_hidden_layers": 0},
            loss_function_str="SPOPlus",
        )

        self.assertAlmostEqual(decision_maker.loss_function.solve_ratio, 0.0)
        self.assertIsNotNone(problem.solution_pool)
        self.assertIs(decision_maker.loss_function.solpool, problem.solution_pool)

        pool_size_before = 0 if problem.solution_pool is None else problem.solution_pool.shape[0]

        runner = Runner(
            decision_maker,
            num_epochs=1,
            use_wandb=False,
            main_metric="abs_regret_pyepo",
            val_metrics=["abs_regret_pyepo"],
        )
        runner.run()

        pool_size_after = 0 if problem.solution_pool is None else problem.solution_pool.shape[0]
        self.assertEqual(pool_size_before, pool_size_after)


if __name__ == "__main__":
    unittest.main()
