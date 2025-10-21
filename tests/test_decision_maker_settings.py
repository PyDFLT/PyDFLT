import unittest

from pydflt.concrete_models.grbpy_two_stage_weighted_set_multi_cover import (
    WeightedSetMultiCover,
)
from pydflt.decision_makers.sfge_decision_maker import SFGEDecisionMaker
from pydflt.generate_data_functions import gen_data_wsmc
from pydflt.problem import Problem
from pydflt.runner import Runner

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
predictor_kwargs = {"num_hidden_layers": 0}
predictor_str = "MLP"


class TestDecisionMakerSettings(unittest.TestCase):
    def test_init_OLS_SFGE(self):
        decision_maker = SFGEDecisionMaker(
            problem=problem,
            learning_rate=0.001,
            batch_size=32,
            device_str="cpu",
            loss_function_str="objective",
            predictor_str=predictor_str,
            predictor_kwargs=predictor_kwargs,
            init_OLS=True,
            standardize_predictions=False,
        )
        runner = Runner(decision_maker, num_epochs=5, use_wandb=False)
        runner.run()


if __name__ == "__main__":
    unittest.main()
