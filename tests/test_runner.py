import unittest
from types import SimpleNamespace

import numpy as np
import pytest

from src.runner import Runner


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


# This allows running the tests directly from the file
if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
