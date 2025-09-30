import unittest

import torch

from src.concrete_models.cvxpy_knapsack import CVXPYDiffKnapsackModel
from src.concrete_models.grbpy_two_stage_weighted_set_multi_cover import (
    WeightedSetMultiCover,
)
from src.decision_makers.differentiable_decision_maker import (
    DifferentiableDecisionMaker,
)
from src.decision_makers.lancer_decision_maker import LancerDecisionMaker
from src.decision_makers.sfge_decision_maker import SFGEDecisionMaker
from src.generate_data_functions.generate_data_knapsack import gen_data_knapsack
from src.generate_data_functions.generate_data_wsmc import gen_data_wsmc
from src.problem import Problem
from src.runner import Runner

# Set-up model and data and problem
model = WeightedSetMultiCover(
    num_items=2,
    num_covers=3,
    penalty=5,
    cover_costs_lb=5,
    cover_costs_ub=50,
    seed=5,
    recovery_ratio=0,
)
data_dict = gen_data_wsmc(
    seed=4,
    num_data=50,
    num_features=2,
    num_items=model.num_items,
    degree=1,
    noise_width=0.5,
)
problem = Problem(
    data_dict=data_dict,
    opt_model=model,
    train_ratio=0.4,
    val_ratio=0.2,
    compute_optimal_decisions=True,
    compute_optimal_objectives=True,
)

# Predictor kwargs
predictor_kwargs = {
    "size": 10,
    "num_hidden_layers": 2,
    "activation": "leaky_relu",
    "output_activation": "leaky_relu",
}
predictor_str = "MLP"


class TestDecisionMakersUsingWSMC(unittest.TestCase):
    def test_pfl(self):
        decision_maker = DifferentiableDecisionMaker(
            problem=problem,
            learning_rate=0.001,
            batch_size=32,
            device_str="cpu",
            loss_function_str="mse",
            predictor_str=predictor_str,
            predictor_kwargs=predictor_kwargs,
        )
        runner = Runner(decision_maker, num_epochs=2, use_wandb=False)
        runner.run()

    def test_sfge(self):
        decision_maker = SFGEDecisionMaker(
            problem=problem,
            learning_rate=0.005,
            batch_size=32,
            device_str="cpu",
            loss_function_str="regret",
            predictor_str=predictor_str,
            predictor_kwargs=predictor_kwargs,
            noisifier_kwargs={"sigma_setting": "independent"},
        )
        runner = Runner(decision_maker, num_epochs=2, use_wandb=False)
        runner.run()

    def test_lancer(self):
        decision_maker = LancerDecisionMaker(
            problem=problem,
            batch_size=32,
            device_str="cpu",
            predictor_str=predictor_str,
            predictor_kwargs=predictor_kwargs,
        )

        runner = Runner(decision_maker, num_epochs=2, use_wandb=False)
        runner.run()

    def test_differentiable_objective_loss(self):
        """
        Tests that DifferentiableDecisionMaker computes the loss correctly for both
        minimization and maximization problems when loss_function_str="objective".
        """
        model = WeightedSetMultiCover(
            num_items=2,
            num_covers=3,
            penalty=5,
            cover_costs_lb=5,
            cover_costs_ub=50,
            seed=5,
            recovery_ratio=0,
        )
        data_dict = gen_data_wsmc(
            seed=4,
            num_data=50,
            num_features=2,
            num_items=model.num_items,
            degree=1,
            noise_width=0.5,
        )
        problem = Problem(
            data_dict=data_dict,
            opt_model=model,
        )

        # 1. Test MINIMIZATION problem (the default)
        decision_maker_min = DifferentiableDecisionMaker(
            problem=problem,
            loss_function_str="objective",
            predictor_str=predictor_str,
            predictor_kwargs=predictor_kwargs,
        )

        # Get a single data batch and move it to the correct device
        idx = problem.generate_batch_indices(batch_size=10)[0]
        data_batch = problem.read_data(idx)

        # For simplicity, we use the optimal decisions and true parameters
        decisions_batch = {}
        for key in model.var_names:
            decisions_batch[key] = data_batch[key + "_optimal"]

        # Calculate the loss using the decision maker
        loss_min = decision_maker_min.get_loss(data_batch, decisions_batch, data_batch)

        # Manually calculate the expected objective value
        expected_objective = problem.opt_model.get_objective(data_batch, decisions_batch, data_batch)

        # For a minimization problem, the loss should be exactly the objective
        self.assertTrue(
            torch.allclose(loss_min, expected_objective),
            "Loss for minimization problem should be equal to the objective.",
        )

        # 2. Test MAXIMIZATION problem
        # Create a copy of the problem to avoid affecting other tests
        model = CVXPYDiffKnapsackModel(num_decisions=10, capacity=20)
        data_dict = gen_data_knapsack(num_data=50, num_items=10, seed=5)
        problem = Problem(opt_model=model, data_dict=data_dict)

        decision_maker_max = DifferentiableDecisionMaker(
            problem=problem,
            loss_function_str="objective",
            predictor_str=predictor_str,
            predictor_kwargs=predictor_kwargs,
        )

        idx = problem.generate_batch_indices(batch_size=10)[0]
        data_batch = problem.read_data(idx)

        # For simplicity, we use the optimal decisions and true parameters
        decisions_batch = {}
        for key in model.var_names:
            decisions_batch[key] = data_batch[key + "_optimal"]

        # Calculate loss using the new decision maker on the same data
        loss_max = decision_maker_max.get_loss(data_batch, decisions_batch, data_batch)

        # Manually calculate the objective from the maximization problem
        objective_for_max = problem.opt_model.get_objective(data_batch, decisions_batch, data_batch)

        # For a maximization problem, the loss should be the *negative* objective
        self.assertTrue(
            torch.allclose(loss_max, -objective_for_max),
            "Loss for maximization problem should be equal to the negative objective.",
        )


if __name__ == "__main__":
    unittest.main()
