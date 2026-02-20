import unittest

from pydflt.concrete_models.grbpy_vrp import VehicleRouting
from pydflt.decision_makers.differentiable_decision_maker import DifferentiableDecisionMaker
from pydflt.generate_data_functions import gen_data_traveling_salesperson
from pydflt.problem import Problem
from pydflt.runner import Runner


class TestCaveLoss(unittest.TestCase):
    def test_cave_vrp_runs(self):
        num_nodes = 5
        opt_model = VehicleRouting(
            num_nodes=num_nodes,
            capacity=10,
            num_vehicles=2,
            seed=1,
            scale_demand=0.5,
            time_limit=1.0,
        )

        data_dict = gen_data_traveling_salesperson(
            seed=1,
            num_data=10,
            num_features=3,
            num_nodes=num_nodes,
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
            loss_function_str="cave",
        )
        runner = Runner(decision_maker, num_epochs=5, use_wandb=False)
        runner.run()


if __name__ == "__main__":
    unittest.main()
