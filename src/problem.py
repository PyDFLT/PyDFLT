import copy
import random
from typing import Union

import numpy as np
import torch
from scipy.spatial import distance

from src.abstract_models.base import OptimizationModel
from src.dataset import DFLDataset


class Problem:
    """
    Manages a decision-focused learning (DFL) problem encapsulating the dataset (`DFLDataset`) and an
    optimization model (`OptimizationModel`). It is responsible for:

    - Splitting the data into training, validation, and test sets.
    - Standardizing features if specified.
    - Computing optimal decisions and objectives for the dataset instances,
      potentially using a k-NN robust approach if configured.
    - Providing an interface to access data batches for different modes (train/val/test).
    - Delegating solving, objective calculation, and evaluation tasks to the `OptimizationModel`.

    Attributes:
        mode (Optional[str]): The current operational mode of the problem instance
            (e.g., 'train', 'validation', 'test').
        train_indices (Optional[np.ndarray]): Indices for the training dataset.
        validation_indices (Optional[np.ndarray]): Indices for the validation dataset.
        test_indices (Optional[np.ndarray]): Indices for the test dataset.
        opt_model (OptimizationModel): The optimization model associated with the problem.
        num_vars (int): Number of decision variables in the optimization model.
        num_features (int): Number of features in the input data.
        params_to_predict_shapes (Dict[str, Tuple]): Shapes of the parameters to be predicted
            by a machine learning model, as defined in `opt_model`.
        num_predictions (int): Total number of scalar values to be predicted, derived
            from `params_to_predict_shapes`.
        dataset (DFLDataset): The dataset object holding all problem instances.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        compute_optimal_decisions (bool, optional): If True, computes and stores the
            optimal decisions for each instance in the dataset. Defaults to True.
        time_respecting_split (bool): Whether to perform a time-respecting split (no shuffling before splitting).
        knn_robust_loss (int, optional): If greater than 0, enables k-NN robust loss
            computation for the training set, using this value as the number of neighbours 'k'.
            This involves finding k-nearest neighbors to perturb parameters for robust
            decision-making. Defaults to 0 (disabled).
        mode_to_indices (dict[str, np.ndarray]): A dictionary mapping mode strings
            ('train', 'validation', 'test') to their corresponding data indices.
        total_num_samples (int): Total number of samples in the dataset.
        all_indices (np.ndarray): An array of all indices from 0 to num_samples-1.
        train_size (int): Number of samples in the training set.
        validation_size (int): Number of samples in the validation set.
        test_size (int): Number of samples in the test set.
    """

    mode: str | None = None
    train_indices: np.ndarray | None = None
    validation_indices: np.ndarray | None = None
    test_indices: np.ndarray | None = None

    def __init__(
        self,
        data_dict: dict[str, np.ndarray],
        opt_model: OptimizationModel,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        compute_optimal_decisions: bool = True,
        compute_optimal_objectives: bool = True,
        standardize_features: bool = False,
        time_respecting_split: bool = False,
        knn_robust_loss: int = 0,
        knn_robust_loss_weight: float = 0.0,
        seed: int | None = None,
        verbose: bool = True,
    ):
        """
        Initializes the Problem instance.

        Args:
            data_dict (dict[str, np.ndarray]): A dictionary where keys are string identifiers
                (e.g., 'features', 'costs') and values are NumPy arrays. It *must* include a 'features' key.
            opt_model (OptimizationModel): An instance of an optimization model that defines
                the problem structure, parameters to predict, and solving logic.
            train_ratio (float, optional): Proportion of the dataset to allocate for training. Defaults to 0.7.
            val_ratio (float, optional): Proportion of the dataset to allocate for validation. Defaults to 0.15.
                The test ratio is defined as 1 - train_ratio - val_ratio.
            compute_optimal_decisions (bool, optional): If True, computes and stores the
                optimal decisions for each instance in the dataset. Defaults to True.
            compute_optimal_objectives (bool, optional): If True, computes and stores the
                optimal objective values for each instance. Requires `compute_optimal_decisions`
                to be True if not using k-NN robust loss. Defaults to True.
            standardize_features (bool, optional): If True, standardizes the 'features' data
                (mean 0, std 1). Defaults to False.
            time_respecting_split (bool, optional): If True, data is split chronologically
                without shuffling. Useful for time-series data. Defaults to False.
            knn_robust_loss (int, optional): If greater than 0, enables k-NN robust loss
                computation for the training set, using this value as the number of neighbours 'k'.
                This involves finding k-nearest neighbors to perturb parameters for robust
                decision-making. Defaults to 0 (disabled).
            knn_robust_loss_weight (float, optional): The weight used when combining
                an instance's original parameters with its neighbors' parameters for k-NN
                robust loss. `perturbed_param = weight * original_param + (1 - weight) * neighbor_param`.
                Defaults to 0.0.
            seed (int | None, optional): Seed for random number generators (NumPy and Python's random)
                to ensure reproducibility in data splitting and other stochastic processes.
                Defaults to None (no explicit seed set).
            verbose (bool): If True, print status messages to the console.

        """
        assert "features" in data_dict, "Data dictionary must contain features key."
        assert isinstance(data_dict["features"], np.ndarray), "Specified features must be a NumPy array."
        assert data_dict["features"].ndim == 2, "Features have to be 2 dimensional"

        # Save optimization model related things
        self.opt_model = opt_model
        self.num_vars = self.opt_model.num_vars
        self.num_features = data_dict["features"].shape[1]
        self.params_to_predict_shapes = self.opt_model.param_to_predict_shapes
        self.num_predictions = self.opt_model.num_predictions

        # Wrap the data into the dataset class
        self.dataset = DFLDataset(data_dict)

        # Set other attributes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.time_respecting_split = time_respecting_split
        self.compute_optimal_decisions = compute_optimal_decisions
        self.knn_robust_loss = knn_robust_loss
        self.verbose = verbose

        # standardize features
        if standardize_features:
            self._standardize_features()

        # Add optimal decisions before splitting
        if compute_optimal_decisions and knn_robust_loss == 0:
            self._add_optimal_decisions(compute_optimal_objectives)

        # Take care of the train/val/test splits
        self.mode_to_indices = {}
        self.total_num_samples = self.dataset.num_samples
        self.all_indices = np.arange(self.total_num_samples)
        self.train_size = int(self.total_num_samples * self.train_ratio)
        self.validation_size = int(self.total_num_samples * self.val_ratio)
        self.test_size = self.total_num_samples - self.train_size - self.validation_size
        assert (
            self.train_size + self.validation_size + self.test_size == self.total_num_samples
        ), "Train, validation, and test sizes should add up to total number of samples!"
        assert self.train_size > 0, "Train set cannot be empty."

        self.split_dataset(seed)

        if knn_robust_loss > 0:
            self._add_optimal_decisions_knn(compute_optimal_objectives, knn_robust_loss, knn_robust_loss_weight)

    def _print_message(self, message: str) -> None:
        """
        Prints a message to the console if verbose mode is enabled.

        Args:
            message (str): The message to be printed.
        """
        if self.verbose:
            print(message)

    def _add_optimal_decisions(self, save_optimal_objectives: bool) -> None:
        """
        Computes optimal decisions and, optionally, their objective values for all instances in the dataset.
        The results are added to `self.dataset`. It solves the optimization problem for each instance using its
        true parameters as provided in `self.dataset.data_dict`.

        Args:
            save_optimal_objectives (bool): If True, also computes and stores the objective
                values corresponding to these optimal decisions.
        """
        self._print_message("Computing optimal decisions for the entire dataset...")
        optimal_decisions = self.solve(self.dataset.data_dict)
        self.dataset.add_data("optimal", optimal_decisions)
        self._print_message("Optimal decisions computed and added to dataset.")
        if save_optimal_objectives:
            self._print_message("Computing optimal objectives for the entire dataset...")
            optimal_objectives = self.get_objective(self.dataset.data_dict, optimal_decisions)
            self.dataset.add_data("objective_optimal", optimal_objectives)
            self._print_message("Optimal objectives computed and added to dataset.")

    def _add_optimal_decisions_knn(self, save_optimal_objectives: bool, k: int, weight: float) -> None:
        """
        Computes k-NN robust optimal decisions for the training set and 'normal' optimal
        decisions for validation and test sets. Optionally computes corresponding objectives.
        Results are added to `self.dataset`.

        For each training instance, this method identifies its k-nearest neighbors based on
        features. It then creates `k` perturbed versions of the instance's optimization parameters
        by blending them with each neighbor's parameters. The optimization problem is solved
        for these `k` perturbed scenarios, and the resulting decisions are averaged to get
        the k-NN robust optimal decision for the original training instance. Reference:
        Robust Losses for Decision-Focused Learning. N. Schutte, K. Postek, N. Yorke-Smith.
        https://doi.org/10.24963/ijcai.2024/538

        Args:
            save_optimal_objectives (bool): If True, computes and stores the objective values.
                For training instances, this is the average objective over the k perturbed scenarios.
            k (int): The number of nearest neighbors to consider for robustness.
            weight (float): The weight for combining original and neighbor parameters.
                           `perturbed = weight * original + (1 - weight) * neighbor`.
        """
        # First check if there is enough data
        num_training_data = len(self.dataset[self.train_indices]["features"])
        assert num_training_data > k, f"When non-zero, knn_robust_loss ({k}) needs to be greater than num_training_data ({num_training_data})"

        self._print_message(f"Computing k-NN robust optimal decisions (k={k}, weight={weight})...")
        # First we add optimal decisions for validation and test
        val_test_indices = np.concatenate((self.validation_indices, self.test_indices))
        assert len(val_test_indices) > 0, "Validation and test sets cannot both be empty when using k-NN robust loss."
        self._print_message("Computing optimal decisions for validation and test sets...")
        val_test_data_batch = self.dataset[val_test_indices]
        optimal_decisions_val_test = self.solve(val_test_data_batch)
        if save_optimal_objectives:
            optimal_objectives_val_test = self.get_objective(val_test_data_batch, optimal_decisions_val_test)
        else:
            optimal_objectives_val_test = None

        # Now we add optimal decisions for train
        self._print_message("Processing training set for k-NN robust losses...")
        train_data_batch = self.dataset[self.train_indices]
        distances = distance.cdist(train_data_batch["features"], train_data_batch["features"], "euclidean")
        indexes = distances.argpartition(k)[:, :k]

        param_to_predict_knn_dict = {}
        for param_to_predict in self.opt_model.param_to_predict_names:
            param_to_predict_knn_dict[param_to_predict] = torch.zeros(
                (
                    len(self.train_indices),
                    *self.opt_model.param_to_predict_shapes[param_to_predict],
                    k,
                ),
                dtype=torch.float32,
            )

        # Find k nearest neighbours
        for i in range(train_data_batch["features"].shape[0]):
            knns = indexes[i, :]
            random.shuffle(knns)
            for (
                param_to_predict,
                param_to_predict_knn,
            ) in param_to_predict_knn_dict.items():
                param_to_predict_knn[i, :, :] = (
                    weight * train_data_batch[param_to_predict][i, :].reshape((-1, 1)) + (1 - weight) * train_data_batch[param_to_predict][knns, :].T
                )

        # Get k optimal decisions for the whole training dataset
        train_dataset = copy.deepcopy(train_data_batch)
        optimal_decisions_list = []
        optimal_objectives_list = []
        for i in range(k):
            for param_to_predict in self.opt_model.param_to_predict_names:
                train_dataset[param_to_predict] = param_to_predict_knn_dict[param_to_predict][:, :, i]
            optimal_decisions_i = self.solve(train_dataset)
            optimal_decisions_list.append(optimal_decisions_i)
            if save_optimal_objectives:
                optimal_objectives_list.append(self.get_objective(train_dataset, optimal_decisions_i))

        # Compute the average
        optimal_decisions_train = {}
        optimal_objectives_train = torch.zeros(len(self.train_indices), dtype=torch.float32)
        for i in range(k):
            if save_optimal_objectives:
                optimal_objectives_train += optimal_objectives_list[i] / k
            for key, decisions in optimal_decisions_list[i].items():
                if key in optimal_decisions_train:
                    optimal_decisions_train[key] += decisions / k
                else:
                    optimal_decisions_train[key] = decisions / k

        if save_optimal_objectives:
            optimal_objectives_total = torch.zeros(self.dataset.num_samples, dtype=torch.float32)
            optimal_objectives_total[self.train_indices] = optimal_objectives_train
            optimal_objectives_total[torch.from_numpy(val_test_indices).long()] = optimal_objectives_val_test
            self.dataset.add_data("objective_optimal", optimal_objectives_total)

        decisions_dict = {}
        for name in self.opt_model.var_names:
            var_shape = self.opt_model.var_shapes[name]
            decisions_dict[name] = torch.zeros((self.dataset.num_samples, *var_shape), dtype=torch.float32)
            decisions_dict[name][self.train_indices] = optimal_decisions_train[name]
            decisions_dict[name][val_test_indices] = optimal_decisions_val_test[name]
        self.dataset.add_data("optimal", decisions_dict)

        self._print_message("k-NN robust decisions (train) and 'normal' optimal decisions (val/test) added to dataset.")

    def split_dataset(self, seed: int | None = None) -> None:
        """
        Splits the dataset indices into training, validation, and test sets.

        The split is based on `self.train_ratio` and `self.val_ratio`.
        If `self.time_respecting_split` is False, indices are shuffled before splitting.
        Sets `self.train_indices`, `self.validation_indices`, `self.test_indices`,
        and populates `self.mode_to_indices`.

        Args:
            seed (int | None, optional): A seed for NumPy's random number generator
                to ensure reproducible shuffling and splitting. If None, the global
                random state is used. Defaults to None.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if not self.time_respecting_split:
            self._print_message("Shuffling indices before splitting...")
            np.random.shuffle(self.all_indices)

        self.train_indices = self.all_indices[: self.train_size]
        self.validation_indices = self.all_indices[self.train_size : self.train_size + self.validation_size]
        self.test_indices = self.all_indices[self.train_size + self.validation_size :]

        self.mode_to_indices["train"] = self.train_indices
        self.mode_to_indices["validation"] = self.validation_indices
        self.mode_to_indices["test"] = self.test_indices

        self._print_message(
            f"Dataset split completed: Train={len(self.train_indices)}, Validation={len(self.validation_indices)}, " f"Test={len(self.test_indices)}"
        )

    def _standardize_features(self) -> None:
        """
        Standardizes the 'features' in the dataset to have zero mean and unit standard deviation.
        Standardization statistics (mean, std) are computed across all samples in the dataset.
        The 'features' field in `self.dataset` is updated in-place.
        """
        feature_data = self.dataset[:]["features"]
        feature_data_float = feature_data.float()
        means = feature_data_float.mean(dim=0)
        stds = feature_data_float.std(dim=0)
        stds[stds == 0] = 1e-8  # Ensure no division by 0
        standardized = (feature_data_float - means) / stds
        self.dataset.add_data("features", standardized)
        self._print_message("Features standardized (mean 0, std 1).")

    def set_mode(self, mode: str) -> None:
        """
        Sets the current operational mode of the Problem instance. This determines which subset of data
        (train, validation, or test) is used by methods like `generate_batch_indices`.

        Args:
            mode (str): The mode to set. Must be one of 'train', 'validation', or 'test'.
        """
        assert self.train_indices is not None, "Split dataset before setting mode!"
        assert mode in [
            "train",
            "validation",
            "test",
        ], "Mode must be train, validation, or test!"
        self.mode = mode
        self._print_message(f"Problem mode set to: {self.mode}")

    def generate_batch_indices(self, batch_size: int) -> list[np.ndarray]:
        """
        Generates a list of index arrays (batches) for the currently set mode. Not that for the 'train' mode,
        the indices are shuffled before being divided into batches. The last batch may be smaller if the number of
        indices is not perfectly divisible by `batch_size`.

        Args:
            batch_size (int): The desired number of samples per batch.

        Returns:
            list[np.ndarray]: A list of NumPy arrays, where each array contains the indices for one batch.
                Returns an empty list if the current mode has no indices.
        """
        assert self.train_indices is not None, "Split dataset before trying to sample!"
        assert self.mode in [
            "train",
            "validation",
            "test",
        ], "Set mode to train, validation, or test before sampling!"
        if self.mode == "train":
            indices_to_use = np.copy(self.train_indices)
            np.random.shuffle(indices_to_use)
        elif self.mode == "validation":
            indices_to_use = np.copy(self.validation_indices)
        else:
            indices_to_use = np.copy(self.test_indices)

        batch_indices = [indices_to_use[i : i + batch_size] for i in range(0, len(indices_to_use), batch_size)]

        return batch_indices

    def read_data(self, idx: Union[int, slice, list[int], np.ndarray]) -> dict[str, torch.Tensor]:
        """
        Retrieves data from the dataset for the specified indices. This is a direct wrapper around the
        `DFLDataset`'s `__getitem__` method.

        Args:
            idx (int | slice | list[int] | np.ndarray): The index, slice, or list/array of indices for which to
            retrieve data.

        Returns:
            dict[str, torch.Tensor]: A dictionary where keys are data field names (e.g., 'features', 'costs') and
                values are the corresponding torch.Tensor data for the requested indices.
        """
        return self.dataset[idx]

    def solve(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Solves the optimization problem for a given batch of data. This method delegates the actual solving process
        to the `solve_batch` method of the `self.opt_model` (OptimizationModel).

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing the data required
                by the optimization model to define the problem instances for the batch.
                Keys are parameter names (e.g., 'costs') and values are torch.Tensors.

        Returns:
            dict[str, torch.Tensor]: A dictionary where keys are decision variable names
                and values are torch.Tensors representing the optimal decisions for the batch.
        """
        decisions_dict = self.opt_model.solve_batch(data_batch)
        return decisions_dict

    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Calculates the objective value for a batch, given the true data, decisions,
        and optionally, predicted parameters. Delegates to `self.opt_model.get_objective`.

        Args:
            data_batch (dict[str, torch.Tensor]): The ground truth parameters for the batch.
            decisions_batch (dict[str, torch.Tensor]): The decisions for which to calculate
                the objective values.
            predictions_batch (dict[str, torch.Tensor] | None, optional): If provided,
                these are the (predicted) parameters that might have been used to *obtain*
                `decisions_batch`. The objective, however, is always evaluated using the
                true parameters from `data_batch`. Defaults to None.

        Returns:
            torch.Tensor: A tensor containing the objective values for each instance in the batch.
        """
        device = list(decisions_batch.values())[0].device
        value = self.opt_model.get_objective(
            {key: val.to(device) for key, val in data_batch.items()},
            decisions_batch,
            predictions_batch=predictions_batch,
        )
        return value

    def evaluate(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Evaluates various performance metrics for a batch, given true data, decisions, and optionally,
        predicted parameters. Delegates to the `evaluate` method of `self.opt_model`.

        Args:
            data_batch (dict[str, torch.Tensor]): The ground truth parameters for the batch.
            decisions_batch (dict[str, torch.Tensor]): The decisions to be evaluated.
            predictions_batch (dict[str, torch.Tensor] | None, optional): Predicted parameters
                that might have been used to obtain `decisions_batch`. Defaults to None.
            metrics (list[str] | None): List of metrics on which to evaluate.

        Returns:
            dict[str, torch.Tensor]: A dictionary where keys are metric names and values are torch.Tensors
                representing the metric values for the batch.
        """
        device = list(decisions_batch.values())[0].device
        eval_dict = self.opt_model.evaluate(
            {key: val.to(device) for key, val in data_batch.items()},
            decisions_batch,
            predictions_batch=predictions_batch,
            metrics=metrics,
        )

        if metrics is not None and "mse" in metrics:  # Add the mean squared error here if wanted
            mse_loss_func = torch.nn.MSELoss()
            batch_size = data_batch["features"].shape[0]
            losses_list = []
            for i in range(batch_size):
                sum_losses = 0
                for key in predictions_batch:
                    true_values = data_batch[key].to(torch.float).to(device)[i]
                    losses = mse_loss_func(predictions_batch[key][i], true_values)
                    sum_losses += losses
                losses_list.append(sum_losses.cpu().detach().numpy().astype(np.float32))
            eval_dict["mse"] = np.array(losses_list)

        return eval_dict
