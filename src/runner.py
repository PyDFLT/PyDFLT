import copy
from typing import Any, Optional

import numpy as np
import optuna
from optuna.trial import Trial

from src.decision_makers import DecisionMaker
from src.logger import Logger
from src.utils.reproducability import set_seeds


class Runner:
    """
    The Runner class allows for running experiments. It uses the given DecisionMaker to run the experiments with and
    initializes a Logger to log results. It handles the training and evaluation process, including logging,
    early stopping, and saving the best models found so far.

    Attributes:
        decision_maker (DecisionMaker): The decision maker instance to be trained and evaluated.
        num_epochs (int): The total number of epochs for training.
        config (Optional[dict[str, Any]]): A dictionary containing experiment configurations to be logged.
        experiments_folder (str): The root directory where experiment results will be saved.
        use_wandb (bool): Flag indicating if Weights & Biases logging is enabled.
        experiment_name (str): Name of the current experiment.
        project_name (str): Name of the project for logging purposes.
        early_stop (bool): Flag indicating if early stopping is enabled.
        min_delta_early_stop (Optional[float]): Minimum change in the main metric to be considered an improvement
                                                for early stopping.
        patience_early_stop (Optional[float]): Number of epochs to wait for an improvement before stopping early.
        save_best (bool): Flag indicating if the best performing model should be saved.
        main_metric (str): The primary metric used for model selection and early stopping.
        store_min_and_max (bool): Whether to store min and max values of metrics in the logger.
        verbose (bool): If True, print status messages to the console.
        main_metric_sense (str): Direction of optimisation for the main metric, either 'MIN' or 'MAX'.
        no_improvement_count (int): Counter for epochs without significant improvement, used for early stopping.
        best_val_metric (float): Stores the best validation metric value achieved so far.
        logger (Logger): The logger instance used for recording experiment results.
    """

    allowed_metrics: list[str] = [
        "objective",
        "abs_regret",
        "rel_regret",
        "sym_rel_regret",
        "mse",
    ]

    def __init__(
        self,
        decision_maker: DecisionMaker,
        num_epochs: int = 3,
        experiments_folder: str = "results/",
        main_metric: str = "abs_regret",
        val_metrics: list[str] = None,
        test_metrics: list[str] = None,
        store_min_and_max: bool = False,
        use_wandb: bool = True,
        experiment_name: str = "-",
        project_name: str = "-",
        early_stop: bool = False,
        min_delta_early_stop: float = None,
        patience_early_stop: float = None,
        save_best: bool = True,
        seed: Optional[int] = None,
        full_reproducibility_GPUs: bool = False,
        config: Optional[dict[str, Any]] = None,
        verbose: bool = True,
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
            min_delta_early_stop (Optional[float]): The minimum change in the main metric to qualify as an improvement.
                                                    A smaller value means more sensitivity.
            patience_early_stop (Optional[float]): Number of epochs with no improvement after which training is stopped.
            save_best (bool): Flag to save the best performing model based on the main_metric on the validation set.
            seed (Optional[int]): Seed for random number generators to ensure reproducibility.
            full_reproducibility_GPUs (bool): Flag to enable/disable GPU reproducibility.
            config (Optional[dict[str, Any]]): A dictionary containing experiment configurations to be logged.
            verbose (bool): If True, print status messages to the console.
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
        self.save_best = save_best
        self.main_metric = main_metric
        self.store_min_and_max = store_min_and_max
        self.verbose = verbose
        self.no_improvement_count = 0
        # self.use_logging = use_logging

        if self.early_stop:
            assert (
                min_delta_early_stop is not None and patience_early_stop is not None
            ), "If early_stop is True, min_delta_early_stop and patience_early_stop are required."

        # State variables
        if self.main_metric == "objective" and self.decision_maker.problem.opt_model.model_sense == "MAX":
            self.main_metric_sense = "MAX"
            self.best_val_metric = -np.inf
        else:
            self.main_metric_sense = "MIN"
            self.best_val_metric = np.inf

        self.val_metrics = self.allowed_metrics if val_metrics is None else val_metrics
        self.test_metrics = self.allowed_metrics if test_metrics is None else test_metrics

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

    def run(self, optuna_trial: Optional[Trial] = None) -> float:
        """
        Runs the experiment. This method iterates through epochs, performs training and validation,
        logs results, handles Optuna trial reporting and pruning, implements early stopping, and saves the best model.
        Finally, it evaluates the best model on the test set.

        Args:
            optuna_trial (Optional[Trial]): An Optuna trial object. If provided, the method will report validation
                                            metrics to Optuna and check for pruning.
        """
        # Note: All *_epoch_results are lists of dictionaries. The length of the list is the number of batches
        # executed during the epoch. For each batch, a dictionary stores one FLOAT (not arrays/tensors) per key

        # Initial validation before training (epoch 0)
        self._print_message(f"Epoch 0/{self.num_epochs}: Starting initial validation...")
        validation_epoch_results_initial = self.decision_maker.run_epoch(mode="validation", epoch_num=0, metrics=self.val_metrics)
        initial_validation_eval = self.logger.log_epoch_results(validation_epoch_results_initial, epoch_num=0)
        best_validation_results = copy.deepcopy(initial_validation_eval)

        if self.save_best:
            self.decision_maker.save_best_predictor()
            self._print_message(f"Initial best validation metric ({self.main_metric}): {initial_validation_eval}")

        self._print_message("Starting training...")
        for epoch in range(1, self.num_epochs + 1):
            self._print_message(f"Epoch: {epoch}/{self.num_epochs}")

            # Training epoch
            train_epoch_results = self.decision_maker.run_epoch(mode="train", epoch_num=epoch)
            self.logger.log_epoch_results(train_epoch_results, epoch_num=epoch)

            # Validation epoch
            validation_epoch_results = self.decision_maker.run_epoch(mode="validation", epoch_num=epoch, metrics=self.val_metrics)
            validation_eval = self.logger.log_epoch_results(validation_epoch_results, epoch_num=epoch)
            self._print_message(f"Validation evaluation ({self.main_metric}): {validation_eval}")

            # Optuna integration
            if optuna_trial is not None:
                optuna_trial.report(validation_eval, epoch)
                # Handle pruning based on intermediate values
                if optuna_trial.should_prune():
                    self._print_message(f"Optuna trial pruned at epoch {epoch}.")
                    raise optuna.TrialPruned()

            # Early stopping
            if self.early_stop:
                if self._check_early_stopping(validation_eval):
                    self._print_message(f"Early stopping triggered at epoch {epoch}!")
                    break

            # Saving best model found so far
            if self.save_best:
                if validation_eval < best_validation_results:
                    self._print_message(f"New best validation evaluation ({self.main_metric}): {validation_eval} " f"(was {best_validation_results})")
                    self.decision_maker.save_best_predictor()
                    best_validation_results = copy.deepcopy(validation_eval)

        # Test results
        self._print_message("Training finished. Evaluating on the test set...")
        if self.save_best:
            self.decision_maker.predictor = self.decision_maker.best_predictor

        test_epoch_results = self.decision_maker.run_epoch(mode="test", epoch_num=self.num_epochs, metrics=self.test_metrics)
        self.logger.log_epoch_results(test_epoch_results, epoch_num=self.num_epochs)
        self.logger.finish()

        return best_validation_results

    def _check_early_stopping(self, current_val_metric: float) -> bool:
        """
        Checks if the early stopping criteria are met.

        Args:
            current_val_metric (float): The validation metric value for the current epoch.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        if self.main_metric_sense == "MAX":
            early_stop_condition_holds = current_val_metric > self.best_val_metric + self.min_delta_early_stop
        else:
            early_stop_condition_holds = current_val_metric < self.best_val_metric + self.min_delta_early_stop

        if early_stop_condition_holds:
            self.best_val_metric = current_val_metric  # Update the best validation metric
            self.no_improvement_count = 0  # Reset the no improvement counter
        else:  # No significant improvement
            self.no_improvement_count += 1  # Increment the counter if no improvement

        # Stop if there has been no improvement for 'patience' epochs
        if self.no_improvement_count >= self.patience_early_stop:
            self._print_message(f"Early stopping condition met: No improvement for {self.no_improvement_count} epochs.")
            return True  # Trigger early stopping

        return False
