import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import wandb


class Logger:
    """
    A comprehensive logging class for machine learning experiments, integrating with Weights & Biases
    and managing local log files. It aggregates per-batch results into epoch-level metrics. See src.utils for method
    load_log for loading logs.

    Attributes:
        experiment_name (str): The name of the current experiment.
        project_name (str): The name of the project.
        config (dict[str, Any] | None): Configuration dictionary for WandB.
        path_to_log (Path): The Path object pointing to the directory where logs are stored.
        log_file_path (Path): The Path object pointing to the text log file.
        store_min_and_max (bool): Flag indicating whether to store min and max for metrics.
        use_wandb (bool): Flag indicating whether Weights & Biases is being used.
        epoch_metrics_list (list[dict[str, float]]): A list storing dictionaries of aggregated
                                                      metrics for each epoch.
        logging_keys (set[str]): A set of all metric keys encountered across epochs.
        main_metric (str): The name of the main metric used for returning values from `log_epoch_results`.
    """

    def __init__(
        self,
        experiment_name: str,
        project_name: str,
        config: Optional[dict[str, Any]] = None,
        use_wandb: bool = True,
        experiments_folder: str = "results/",
        main_metric: str = "abs_regret",
        store_min_and_max: bool = True,
    ):
        """
        Initializes the Logger with experiment details and logging configurations.

        Args:
            experiment_name (str): The name of the current experiment.
            project_name (str): The name of the project, used for grouping experiments in WandB.
            config (dict[str, Any] | None): A dictionary of configuration parameters for the experiment.
                                        This will be logged to Weights & Biases. Defaults to None.
            use_wandb (bool): If True, initializes and uses Weights & Biases for logging. Defaults to True.
            experiments_folder (str): The base folder where experiment results and logs will be stored.
                                      Defaults to 'results/'.
            main_metric (str): The key of the main metric to be returned from `log_epoch_results`
                                after aggregation. Defaults to 'abs_regret'.
            store_min_and_max (bool): If True, stores the min and max values for each aggregated metric
                                      along with the mean. Defaults to True.
        """

        self.experiment_name = experiment_name
        self.project_name = project_name
        self.config = config
        self.path_to_log = Path(experiments_folder) / experiment_name / project_name
        self.log_file_path = self.path_to_log / "log.txt"  # Path to the text log file
        self.store_min_and_max = store_min_and_max

        # Ensure the logs directory exists
        os.makedirs(self.path_to_log, exist_ok=True)

        self.use_wandb = use_wandb
        self.epoch_metrics_list = []
        self.logging_keys = set()
        self.main_metric = main_metric
        if use_wandb:
            self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        """
        Initializes the Weights & Biases run. This method is called internally if `use_wandb` is True.
        """
        wandb.init(project=self.project_name, name=self.experiment_name, config=self.config)

    def log_epoch_results(self, per_batch_results: list[dict[str, float]], epoch_num: int) -> float | None:
        """
        Receives a list of dictionaries with per-batch results, aggregates them, and logs.
        Keys in the per-batch results are typically formatted as 'mode/name', where 'mode'
        is 'train', 'val', or 'test', and 'name' is the metric name.
        Results are aggregated (mean, min, max) and logged to a local file and optionally to WandB.

        Args:
            per_batch_results (list[dict[str, float]]): A list of dictionaries, where each dictionary
                                                         contains metric names and their corresponding
                                                         values for a single batch.
            epoch_num (int): The current epoch number.

        Returns:
            float | None: The aggregated mean value of the `main_metric` for the current epoch,
                          if found; otherwise, None.
        """
        # Aggregate results
        epoch_metrics = defaultdict(list)
        for batch in per_batch_results:
            for key, value in batch.items():
                epoch_metrics[key].append(np.mean(value))

        # Calculate aggregation for each metric
        aggregate_epoch_metrics = {}
        for key, value in epoch_metrics.items():
            value = np.array(value)
            aggregate_epoch_metrics[f"{key}_mean"] = value.mean()
            if self.store_min_and_max:
                aggregate_epoch_metrics[f"{key}_max"] = value.max()
                aggregate_epoch_metrics[f"{key}_min"] = value.min()
            # aggregate_epoch_metrics[f'{key}_std'] = value.std()
        self.epoch_metrics_list.append(aggregate_epoch_metrics)

        # Log to Weights & Biases if enabled
        if self.use_wandb:
            wandb.log({f"{k}": v for k, v in aggregate_epoch_metrics.items()}, step=epoch_num)

        # Add new keys to logging_keys
        self.logging_keys.update(aggregate_epoch_metrics.keys())

        # Write metrics to log file
        self._write_to_file(aggregate_epoch_metrics, epoch_num)
        self.printout()
        for key in aggregate_epoch_metrics:
            if key == (f"validation/{self.main_metric}_mean" or key == f"train/{self.main_metric}_mean" or key == f"test/{self.main_metric}_mean"):
                value = aggregate_epoch_metrics[key]

                return value

        return None

    def _write_to_file(self, epoch_metrics: dict[str, float], epoch_num: int) -> None:
        """
        Writes the epoch's aggregated metrics to a text file.

        Args:
            epoch_metrics (dict[str, float]): A dictionary of aggregated metrics for the current epoch.
            epoch_num (int): The current epoch number.
        """
        with open(self.log_file_path, "a") as f:
            f.write(f"Epoch {epoch_num}:\n")
            for key, value in epoch_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")

    def printout(self) -> None:
        """
        Utility function that prints out the aggregated epoch results to the console.
        It calculates the average for each metric across all logged epochs and prints it.
        """
        if self.epoch_metrics_list:
            print("Epoch Results:")
            for key in self.logging_keys:
                values = [metrics[key] for metrics in self.epoch_metrics_list if key in metrics]
                avg_value = sum(values) / len(values) if values else 0
                print(f"{key}: {avg_value:.4f}")
        else:
            print("No results to print.")

    def finish(self) -> None:
        """
        Finishes the Weights & Biases run if it was initialized.
        """
        if self.use_wandb:
            wandb.finish()
