import unittest
from types import SimpleNamespace

import numpy as np
import pytest
from pydflt.concrete_models import CVXPYDiffKnapsackModel
from pydflt.decision_makers.differentiable_decision_maker import DifferentiableDecisionMaker
from pydflt.generate_data_functions import gen_data_knapsack
from pydflt.problem import Problem
from pydflt.runner import Runner


class DummyDecisionMaker:
    def __init__(self, model_sense: str = "MAX") -> None:
        self.problem = SimpleNamespace(opt_model=SimpleNamespace(model_sense=model_sense))
        self.predictor = "current"
        self.best_predictor = "best"

    def run_epoch(self, *args, **kwargs):
        return []

    def save_best_predictor(self) -> None:
        self.best_predictor = self.predictor


def test_runner_maximization_updates_main_metric(tmp_path) -> None:
    decision_maker = DummyDecisionMaker(model_sense="MAX")
    runner = Runner(
        decision_maker=decision_maker,
        num_epochs=1,
        experiments_folder=str(tmp_path),
        main_metric="objective",
        val_metrics=["objective"],
        test_metrics=["objective"],
        use_wandb=False,
        early_stop=True,
        min_delta_early_stop=0.05,
        patience_early_stop=2,
        save_best=False,
    )

    assert runner.main_metric_sense == "MAX"
    assert runner.best_val_metric == -np.inf

    assert runner._check_early_stopping(1.0) is False
    assert runner.best_val_metric == pytest.approx(1.0)
    assert runner.no_improvement_count == 0

    assert runner._check_early_stopping(1.02) is False
    assert runner.best_val_metric == pytest.approx(1.0)
    assert runner.no_improvement_count == 1

    assert runner._check_early_stopping(1.07) is False
    assert runner.best_val_metric == pytest.approx(1.07)
    assert runner.no_improvement_count == 0

    assert runner._check_early_stopping(1.08) is False
    assert runner.best_val_metric == pytest.approx(1.07)
    assert runner.no_improvement_count == 1

    assert runner._check_early_stopping(1.05) is True
    assert runner.best_val_metric == pytest.approx(1.07)
    assert runner.no_improvement_count == 2


def test_runner_empty_validation_uses_train_eval(tmp_path) -> None:
    model = CVXPYDiffKnapsackModel(num_decisions=5, capacity=10)
    data_dict = gen_data_knapsack(num_data=4, num_items=5, seed=1)
    problem = Problem(
        opt_model=model,
        data_dict=data_dict,
        train_ratio=1.0,
        val_ratio=0.0,
        compute_optimal_decisions=True,
        compute_optimal_objectives=True,
    )
    decision_maker = DifferentiableDecisionMaker(
        problem=problem,
        learning_rate=1e-3,
        batch_size=2,
        device_str="cpu",
        predictor_str="MLP",
        predictor_kwargs={"num_hidden_layers": 0},
        loss_function_str="objective",
    )
    runner = Runner(
        decision_maker=decision_maker,
        num_epochs=1,
        experiments_folder=str(tmp_path),
        main_metric="objective",
        val_metrics=["objective"],
        test_metrics=["objective"],
        use_wandb=False,
        early_stop=False,
        save_best=True,
    )

    result = runner.run()
    assert result is not None


def test_runner_relative_threshold_min_metric(tmp_path) -> None:
    decision_maker = DummyDecisionMaker(model_sense="MIN")
    runner = Runner(
        decision_maker=decision_maker,
        num_epochs=1,
        experiments_folder=str(tmp_path),
        main_metric="objective",
        val_metrics=["objective"],
        test_metrics=["objective"],
        use_wandb=False,
        early_stop=True,
        min_delta_early_stop=0.1,
        patience_early_stop=2,
        save_best=False,
    )

    assert runner._check_early_stopping(2.0) is False
    assert runner.best_val_metric == pytest.approx(2.0)

    # Relative threshold = 2.0 - 0.1 * 2.0 = 1.8
    assert runner._check_early_stopping(1.9) is False
    assert runner.best_val_metric == pytest.approx(2.0)

    assert runner._check_early_stopping(1.7) is False
    assert runner.best_val_metric == pytest.approx(1.7)


def test_runner_time_based_early_stop(tmp_path) -> None:
    decision_maker = DummyDecisionMaker(model_sense="MIN")
    runner = Runner(
        decision_maker=decision_maker,
        num_epochs=1,
        experiments_folder=str(tmp_path),
        main_metric="objective",
        val_metrics=["objective"],
        test_metrics=["objective"],
        use_wandb=False,
        early_stop=True,
        min_delta_early_stop=0.01,
        patience_early_stop=None,
        patience_early_stop_seconds=0.1,
        save_best=False,
    )

    runner.best_val_metric = 1.0
    runner.last_improvement_time = runner.start_time = 0.0
    runner.last_improvement_time = runner.last_improvement_time - 10.0

    assert runner._check_early_stopping(0.995) is True


# This allows running the tests directly from the file
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
