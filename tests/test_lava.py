import unittest

from pydflt.concrete_models import GRBPYKnapsackModel
from pydflt.decision_makers.differentiable_decision_maker import DifferentiableDecisionMaker
from pydflt.generate_data_functions import gen_data_knapsack
from pydflt.problem import Problem
from pydflt.runner import Runner


class TestLavaLoss(unittest.TestCase):
    def test_lava_knapsack_runs(self):

        opt_model = GRBPYKnapsackModel(
            num_decisions=6,
            capacity=15,
            weights_lb=3.0,
            weights_ub=8.0,
            seed=1,
            time_limit=1.0,
        )
        data_dict = gen_data_knapsack(
            seed=1,
            num_data=20,
            num_features=3,
            num_items=6,
        )
        problem = Problem(
            data_dict=data_dict,
            opt_model=opt_model,
            train_ratio=0.6,
            val_ratio=0.2,
            compute_optimal_decisions=True,
            compute_optimal_objectives=True,
        )

        predictor_kwargs = {
            "num_hidden_layers": 0,
        }
        decision_maker = DifferentiableDecisionMaker(
            problem=problem,
            learning_rate=1e-3,
            batch_size=5,
            device_str="cpu",
            predictor_str="MLP",
            predictor_kwargs=predictor_kwargs,
            loss_function_str="lava",
        )
        runner = Runner(decision_maker, num_epochs=5, use_wandb=False)
        runner.run()


if __name__ == "__main__":
    unittest.main()
