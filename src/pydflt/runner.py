import copy
import time
from typing import Any, ClassVar

import numpy as np
import optuna
from optuna.trial import Trial

from pydflt.decision_makers import DecisionMaker
from pydflt.logger import Logger
from pydflt.utils.reproducibility import set_seeds


class Runner:
    """
    The Runner class allows for running experiments. It uses the given DecisionMaker to run the experiments with and
    initializes a Logger to log results. It handles the training and evaluation process, including logging,
    early stopping, and saving the best models found so far.

    Attributes:
        decision_maker (DecisionMaker): The decision maker instance to be trained and evaluated.
        num_epochs (int): The total number of epochs for training.
        config (dict[str, Any] | None): A dictionary containing experiment configurations to be logged.
        experiments_folder (str): The root directory where experiment results will be saved.
        use_wandb (bool): Flag indicating if Weights & Biases logging is enabled.
        experiment_name (str): Name of the current experiment.
        project_name (str): Name of the project for logging purposes.
        early_stop (bool): Flag indicating if early stopping is enabled.
        min_delta_early_stop (float | None): Minimum change in the main metric to be considered an improvement
                                              for early stopping.
        patience_early_stop (float | None): Number of epochs to wait for an improvement before stopping early.
        save_best (bool): Flag indicating if the best performing model should be saved.
        main_metric (str): The primary metric used for model selection and early stopping.
        store_min_and_max (bool): Whether to store min and max values of metrics in the logger.
        verbose (bool): If True, print status messages to the console.
        main_metric_sense (str): Direction of optimisation for the main metric, either 'MIN' or 'MAX'.
        no_improvement_count (int): Counter for epochs without significant improvement, used for early stopping.
        best_val_metric (float): Stores the best validation metric value achieved so far.
        logger (Logger): The logger instance used for recording experiment results.
    """

    allowed_metrics: ClassVar[list[str]] = [
        "objective",
        "abs_regret",
        "rel_regret",
        "sym_rel_regret",
        "mse",
        "mae",
        "used_loss",
        "mse_norm",
        "mae_norm",
    ]
    default_metrics: ClassVar[list[str]] = [
        "objective",
        "abs_regret",
        "rel_regret",
        "sym_rel_regret",
        "mse",
        "mae",
    ]

    def __init__(
        self,
        decision_maker: DecisionMaker,
        num_epochs: int = 3,
        experiments_folder: str = "results/",
        main_metric: str = "abs_regret",
        val_metrics: list[str] | None = None,
        test_metrics: list[str] | None = None,
        store_min_and_max: bool = False,
        use_wandb: bool = False,
        experiment_name: str = "-",
        project_name: str = "-",
        early_stop: bool = False,
        min_delta_early_stop: float | None = None,
        patience_early_stop: float | None = None,
        patience_early_stop_seconds: float | None = None,
        save_best: bool = True,
        seed: int | None = None,
        full_reproducibility_GPUs: bool = False,
        config: dict[str, Any] | None = None,
        verbose: bool = True,
        timeout_seconds: float | None = None,
        start_time: float | None = None,
        # use_logging: bool = True,
    ):
        """
        Initializes a Runner instance.

        Args:
            decision_maker (DecisionMaker): The decision maker to be trained and evaluated.
            num_epochs (int): The total number of epochs for training.
            experiments_folder (str): The root directory where experiment results will be saved.
            main_metric (str): The primary metric used for model selection and early stopping.
            store_min_and_max (bool): Whether to store min and max values of metrics in the logger.
            use_wandb (bool): Flag to enable/disable logging with Weights & Biases.
            experiment_name (str): Name of the current experiment (used for Weights & Biases).
            project_name (str): Name of the project (used for Weights & Biases).
            early_stop (bool): Flag to enable/disable early stopping.
            min_delta_early_stop (float | None): The minimum change in the main metric to qualify as an improvement.
                                                 A smaller value means more sensitivity.
            patience_early_stop (float | None): Number of epochs with no improvement after which training is stopped.
            patience_early_stop_seconds (float | None): Seconds without improvement after which training is stopped.
            save_best (bool): Flag to save the best performing model based on the main_metric on the validation set.
            seed (int | None): Seed for random number generators to ensure reproducibility.
            full_reproducibility_GPUs (bool): Flag to enable/disable GPU reproducibility.
            config (dict[str, Any] | None): A dictionary containing experiment configurations to be logged.
            verbose (bool): If True, print status messages to the console.
            timeout_seconds (float | None): If set, stop training after this many seconds (test still runs).
            start_time (float | None): If set, use this as the run start time for timeouts.
        """

        # Set up the seeds
        if seed is not None:
            set_seeds(seed, full_reproducibility_GPUs)

        # Save input parameters
        self.decision_maker = decision_maker
        self.num_epochs = num_epochs
        self.config = config
        self.experiments_folder = experiments_folder
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.early_stop = early_stop
        self.min_delta_early_stop = min_delta_early_stop
        self.patience_early_stop = patience_early_stop
        self.patience_early_stop_seconds = patience_early_stop_seconds
        self.save_best = save_best
        self.main_metric = main_metric
        self.store_min_and_max = store_min_and_max
        self.verbose = verbose
        self.timeout_seconds = timeout_seconds
        self.start_time = start_time
        self.no_improvement_count = 0
        self.last_improvement_time = None
        # self.use_logging = use_logging

        if self.early_stop:
            has_epoch_patience = patience_early_stop is not None
            has_time_patience = patience_early_stop_seconds is not None
            assert min_delta_early_stop is not None, "If early_stop is True, min_delta_early_stop is required."
            assert has_epoch_patience or has_time_patience, "If early_stop is True, patience_early_stop or patience_early_stop_seconds is required."

        # State variables
        if self.main_metric == "objective" and self.decision_maker.problem.opt_model.model_sense == "MAX":
            self.main_metric_sense = "MAX"
            self.best_val_metric = -np.inf
        else:
            self.main_metric_sense = "MIN"
            self.best_val_metric = np.inf

        self.val_metrics = self.default_metrics if val_metrics is None else val_metrics
        self.test_metrics = self.default_metrics if test_metrics is None else test_metrics

        for metric in self.val_metrics + self.test_metrics:
            assert metric in self.allowed_metrics, f"Metric {metric} has to be from {self.allowed_metrics}."
        assert self.main_metric in self.val_metrics, "val_metrics has to include main_metric, as otherwise main_metric is not recorded."

        # Initialize logger
        self._initialize_logger()

    def _print_message(self, message: str) -> None:
        """
        Prints a message to the console if verbose mode is enabled.

        Args:
            message (str): The message to be printed.
        """
        if self.verbose:
            print(message)

    def _initialize_logger(self) -> None:
        """
        Creates and initializes the Logger instance for the experiment.
        This logger will handle saving metrics and configurations.
        """
        self.logger = Logger(
            experiment_name=self.experiment_name,
            project_name=self.project_name,
            config=self.config,
            use_wandb=self.use_wandb,
            experiments_folder=self.experiments_folder,
            main_metric=self.main_metric,
            store_min_and_max=self.store_min_and_max,
        )

    def run(self, optuna_trial: Trial | None = None) -> float:
        """
        Runs the experiment. This method iterates through epochs, performs training and validation,
        logs results, handles Optuna trial reporting and pruning, implements early stopping, and saves the best model.
        Finally, it evaluates the best model on the test set.

        Args:
            optuna_trial (Trial | None): An Optuna trial object. If provided, the method will report validation
                                          metrics to Optuna and check for pruning.
        """
        # Note: All *_epoch_results are lists of dictionaries. The length of the list is the number of batches
        # executed during the epoch. For each batch, a dictionary stores one FLOAT (not arrays/tensors) per key

        # Initial validation before training (epoch 0)
        if self.start_time is None:
            self.start_time = time.perf_counter()
        regret_metrics = ["abs_regret", "rel_regret", "sym_rel_regret"]
        if any(m in self.val_metrics or m in self.test_metrics for m in regret_metrics) and not getattr(
            self.decision_maker.problem, "compute_optimal_objectives", False
        ):
            self._print_message(
                "Warning: regret metrics requested but compute_optimal_objectives=False. "
                "Optimal objectives will be computed on-the-fly during evaluation, which is slower than precomputing."
            )
        self._print_message(f"Epoch 0/{self.num_epochs}: Starting initial validation...")
        validation_epoch_results_initial = self.decision_maker.run_epoch(mode="validation", epoch_num=0, metrics=self.val_metrics)
        validation_is_empty = len(validation_epoch_results_initial) == 0
        initial_validation_eval = None
        if not validation_is_empty:
            initial_validation_eval = self.logger.log_epoch_results(validation_epoch_results_initial, epoch_num=0)
            if np.isinf(self.best_val_metric) or np.isneginf(self.best_val_metric):
                self.best_val_metric = initial_validation_eval
                self.last_improvement_time = time.perf_counter()
                self.no_improvement_count = 0
        best_validation_results = copy.deepcopy(initial_validation_eval) if initial_validation_eval is not None else None

        if self.save_best and initial_validation_eval is not None:
            self.decision_maker.save_best_predictor()
            self._print_message(f"Initial best validation metric ({self.main_metric}): {initial_validation_eval}")
        if self.last_improvement_time is None:
            self.last_improvement_time = time.perf_counter()

        self._print_message("Starting training...")
        train_eval = None
        for epoch in range(1, self.num_epochs + 1):
            self._print_message(f"Epoch: {epoch}/{self.num_epochs}")

            # Training epoch
            train_epoch_results = self.decision_maker.run_epoch(mode="train", epoch_num=epoch)
            train_eval = self.logger.log_epoch_results(train_epoch_results, epoch_num=epoch)

            if self.timeout_seconds is not None:
                elapsed = time.perf_counter() - self.start_time
                if elapsed >= self.timeout_seconds:
                    self._print_message(f"Timeout reached after {elapsed:.2f}s at epoch {epoch}. Stopping training.")
                    break

            # Validation epoch
            if not validation_is_empty:
                validation_epoch_results = self.decision_maker.run_epoch(mode="validation", epoch_num=epoch, metrics=self.val_metrics)
                validation_eval = self.logger.log_epoch_results(validation_epoch_results, epoch_num=epoch)
                self._print_message(f"Validation evaluation ({self.main_metric}): {validation_eval}")
            else:
                validation_eval = train_eval
                self._print_message(f"Train evaluation ({self.main_metric}): {validation_eval} (validation set empty)")
                if np.isinf(self.best_val_metric) or np.isneginf(self.best_val_metric):
                    self.best_val_metric = validation_eval
                    self.last_improvement_time = time.perf_counter()
                    self.no_improvement_count = 0

            # Optuna integration
            if optuna_trial is not None:
                optuna_trial.report(validation_eval, epoch)
                # Handle pruning based on intermediate values
                if optuna_trial.should_prune():
                    self._print_message(f"Optuna trial pruned at epoch {epoch}.")
                    raise optuna.TrialPruned()

            # Saving best model found so far
            if self.save_best and (best_validation_results is None or validation_eval < best_validation_results):
                self._print_message(f"New best validation evaluation ({self.main_metric}): {validation_eval} " f"(was {best_validation_results})")
                self.decision_maker.save_best_predictor()
                best_validation_results = copy.deepcopy(validation_eval)

            # Early stopping
            if self.early_stop and self._check_early_stopping(validation_eval):
                self._print_message(f"Early stopping triggered at epoch {epoch}!")
                break

        # Test results
        self._print_message("Training finished. Evaluating on the test set...")
        if self.save_best:
            self.decision_maker.predictor = self.decision_maker.best_predictor

        test_epoch_results = self.decision_maker.run_epoch(mode="test", epoch_num=self.num_epochs, metrics=self.test_metrics)
        self.logger.log_epoch_results(test_epoch_results, epoch_num=self.num_epochs)
        self.logger.finish()

        return best_validation_results if best_validation_results is not None else train_eval

    def _check_early_stopping(self, current_val_metric: float) -> bool:
        """
        Checks if the early stopping criteria are met.

        Args:
            current_val_metric (float): The validation metric value for the current epoch.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        if current_val_metric is None:
            return False
        if np.isinf(self.best_val_metric) or np.isneginf(self.best_val_metric):
            self.best_val_metric = current_val_metric
            self.no_improvement_count = 0
            self.last_improvement_time = time.perf_counter()
            return False
        if self.min_delta_early_stop is None:
            min_delta = 0.0
        else:
            min_delta = self.min_delta_early_stop
        if self.main_metric_sense == "MAX":
            threshold = self.best_val_metric + min_delta * abs(self.best_val_metric)
            early_stop_condition_holds = current_val_metric > threshold
        else:
            threshold = self.best_val_metric - min_delta * abs(self.best_val_metric)
            early_stop_condition_holds = current_val_metric < threshold

        if early_stop_condition_holds:
            self.best_val_metric = current_val_metric  # Update the best validation metric
            self.no_improvement_count = 0  # Reset the no improvement counter
            self.last_improvement_time = time.perf_counter()
        else:  # No significant improvement
            self.no_improvement_count += 1  # Increment the counter if no improvement

        # Stop if there has been no improvement for 'patience' epochs
        if self.patience_early_stop is not None and self.no_improvement_count >= self.patience_early_stop:
            self._print_message(f"Early stopping condition met: No improvement for {self.no_improvement_count} epochs.")
            return True  # Trigger early stopping

        if self.patience_early_stop_seconds is not None:
            if self.last_improvement_time is None:
                self.last_improvement_time = time.perf_counter()
            elapsed = time.perf_counter() - self.last_improvement_time
            if elapsed >= self.patience_early_stop_seconds:
                self._print_message(f"Early stopping condition met: No improvement for {elapsed:.2f}s.")
                return True

        return False
