import copy
from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_squared_error
from torch import nn

from src.abstract_models.base import OptimizationModel
from src.noisifier import Noisifier
from src.predictors import IMPLEMENTED_PREDICTORS
from src.predictors.base import Predictor
from src.problem import Problem
from src.utils.reproducability import set_seeds


class DecisionMaker:
    """
    Abstract base class for a decision-making agent in a decision-focused learning pipeline.

    This class orchestrates the interaction between a predictive model (`predictor`),
    an optional `noisifier` to add stochasticity to predictions, and a `decision_model`
    (an optimization model) to make decisions based on (potentially noisy) predictions.
    It handles the initialization of these components, the process of predicting
    parameters and making decisions, and provides hooks for training and evaluation.

    The class is organized into several categories of methods:
    - Abstract methods for subclass implementation: Need to be implemented for a subclass to work.
    - Core logic methods: Methods that include the core logic of what a decision maker entails
    - Helper methods: Local methods that support other methods
    - Conversion methods: Methods that convert, i.e., transform or reshape certain objects.
    - Initialization methods: Methods that do initialization

    Attributes:
        problem (Problem): The problem instance containing data and the true optimization model.
        learning_rate (float): Learning rate for the trainable components.
        device (torch.device): The device (CPU or GPU) on which computations are performed.
        predictor_str (str): String identifier for the type of predictor used.
        decision_model_str (str): String identifier for the type of decision model used.
        loss_function_str (str): String identifier for the loss function.
        to_decision_pars (str): Strategy for converting distributional predictions to
            deterministic parameters for the decision model (e.g., 'sample', 'quantiles').
        use_dist_at_mode (str): Specifies if/when to use full distributional output from
            predictor/noisifier (e.g., 'test' mode).
        use_noisifier (bool): Flag indicating whether a noisifier is used on top of the base predictor.
        standardize_predictions (bool): Flag indicating whether to standardize predictions.
        predictor_kwargs (dict[str, Any]): Keyword arguments for predictor initialization.
        noisifier_kwargs (dict[str, Any]): Keyword arguments for noisifier initialization.
        decision_model_kwargs (dict[str, Any]): Keyword arguments for decision model initialization.
        decision_model (OptimizationModel): The instantiated optimization model used for making decisions.
        predictor (Predictor): The instantiated predictive model.
        best_predictor (Optional[Predictor]): A copy of the predictor with the best validation performance.
        noisifier (Optional[Noisifier]): The instantiated noisifier, if `use_noisifier` is True.
        trainable_predictive_model (nn.Module): The PyTorch module that is trained (either
            the predictor or the noisifier if used).
        num_scenarios (Optional[int]): Number of scenarios used if `to_decision_pars` involves sampling
            or quantiles, derived from `decision_model_kwargs`.
        _solver_calls (int): Counter for the number of times the decision model's solver is called.
        _epoch_counts (int): Counter for the number of epochs run (though not explicitly updated in this snippet).
    """

    allowed_losses: list[str] = [
        # Define allowed loss function strings here, e.g., 'objective', 'regret'
    ]

    allowed_decision_models: list[str] = [
        # Define allowed decision model strings here, e.g., 'base', 'quadratic', 'scenario_based'
    ]

    allowed_predictors: list[str] = [
        # Define allowed predictor strings here, e.g., 'MLP', 'Normal', 'LinearSKL'
    ]

    decision_model: OptimizationModel
    predictor: Predictor
    best_predictor: Optional[Predictor] = None
    noisifier: Optional[Noisifier]
    trainable_predictive_model: nn.Module

    def __init__(
        self,
        problem: Problem,
        learning_rate: float = 1e-4,
        device_str: str = "cpu",
        predictor_str: str = "MLP",
        decision_model_str: str = "base",
        loss_function_str: str = "objective",
        to_decision_pars: str = "none",
        use_dist_at_mode: str = "none",
        use_noisifier: bool = False,
        standardize_predictions: bool = True,
        init_OLS: bool = False,
        seed: Union[int, None] = None,
        predictor_kwargs: dict = None,
        noisifier_kwargs: dict = None,
        decision_model_kwargs: dict = None,
    ):
        """
        Initializes a DecisionMaker instance.

        Sets up the problem, device, learning rate, and configurations for the predictor,
        decision model, and noisifier. It then initializes these components.

        Args:
            problem (Problem): An instance of the `Problem` class, containing data and
                the true optimization model.
            learning_rate (float, optional): Learning rate used for training the
                learnable parts of the decision maker. Defaults to 1e-4.
            device_str (str, optional): Device to use for PyTorch computations (e.g., 'cpu', 'cuda').
                Defaults to 'cpu'.
            predictor_str (str, optional): Identifier for the predictor model to use
                (e.g., 'MLP', 'Normal', 'LinearSKL'). Must be a key in `IMPLEMENTED_PREDICTORS`
                and present in `self.allowed_predictors`. Defaults to 'linear'.
            decision_model_str (str, optional): Identifier for the decision model strategy.
                'base' uses a copy of the problem's optimization model. Other values like
                'quadratic' or 'scenario_based' create variants. Must be in `self.allowed_decision_models`.
                Defaults to 'base'.
            loss_function_str (str, optional): Identifier for the loss function to be used
                during training. Must be in `self.allowed_losses`. Defaults to 'objective'.
            to_decision_pars (str, optional): Strategy to convert distributional predictions
                into parameters for the decision model. Options: 'none', 'sample', 'quantiles'.
                Defaults to 'none'.
            use_dist_at_mode (str, optional): Specifies when to use the full distribution from
                the predictor/noisifier for decision-making. Options: 'none', 'test'.
                'validation' could be a future option. Defaults to 'none'.
            use_noisifier (bool, optional): If True, a `Noisifier` is added on top of the
                base predictor to introduce stochasticity. Defaults to False.
            standardize_predictions (bool, optional): If True, an attempt is made to standardize
                predictions by scaling them based on training data statistics (mean and std).
                Currently implemented mainly for MLP predictors. Defaults to True.
            init_OLS (bool, optional): If True and the predictive model is linear, its parameters are initialized
                using ordinary least squares through sklearn. When the predictor has multiple outputs it uses
                sklearn's quantile regressor to initialize.
            seed (Optional[int], optional): Seed for random number generators (NumPy, Python's random, PyTorch)
                to ensure reproducibility. Defaults to None.
            predictor_kwargs (Optional[dict[str, Any]], optional): Keyword arguments to pass to the
                predictor's constructor. Defaults to None (empty dict).
            noisifier_kwargs (Optional[dict[str, Any]], optional): Keyword arguments to pass to the
                Noisifier's constructor. Defaults to None (empty dict).
            decision_model_kwargs (Optional[dict[str, Any]], optional): Keyword arguments for
                creating variants of the decision model (e.g., for 'quadratic' or 'scenario_based').
                Defaults to None (empty dict).
        """

        assert loss_function_str in self.allowed_losses, "Loss must be from %s" % self.allowed_losses
        assert decision_model_str in self.allowed_decision_models, "Decision model must be from %s" % self.allowed_decision_models
        assert predictor_str in self.allowed_predictors, "Predictor must be from %s" % self.allowed_predictors
        allowed_to_decision_pars_approaches = ["none", "sample", "quantiles"]
        assert to_decision_pars in allowed_to_decision_pars_approaches, "To decision pars must be from %s" % allowed_to_decision_pars_approaches
        use_dist_at_mode_approaches = ["none", "test"]
        assert use_dist_at_mode in use_dist_at_mode_approaches, "Use dist at mode must be from %s" % use_dist_at_mode_approaches
        if decision_model_str == "quadratic" and (standardize_predictions or init_OLS):
            assert problem.compute_optimal_decisions or problem.knn_robust_loss > 0, (
                "When the decision model is quadratic and init_OLS or standardize_predictions is True, "
                "compute_optimal_decisions must be True (or knn_robust_loss must be > 0)."
            )
        assert not (
            use_dist_at_mode != "none" and to_decision_pars == "none"
        ), "When use_dist_at_mode is not 'none', to_decision_pars has to be 'sample' or 'quantiles'."

        if seed is not None:
            set_seeds(seed)

        # Store core attributes
        self.problem = problem
        self.device = torch.device(device_str)
        self.learning_rate = learning_rate

        # Settings
        self.to_decision_pars = to_decision_pars
        self.use_dist_at_mode = use_dist_at_mode
        self.standardize_predictions = standardize_predictions
        self.init_OLS = init_OLS
        self.use_noisifier = use_noisifier

        # Set strings
        self.predictor_str = predictor_str
        self.decision_model_str = decision_model_str
        self.loss_function_str = loss_function_str

        # Store kwargs
        self.predictor_kwargs = predictor_kwargs or {}
        self.noisifier_kwargs = noisifier_kwargs or {}
        self.decision_model_kwargs = decision_model_kwargs or {}

        # Set num_scenarios
        self.num_scenarios = None
        if self.to_decision_pars != "none":
            assert (
                "num_scenarios" in self.decision_model_kwargs
            ), "If to_decision_pars is specified, num_scenarios has to be given in the decision_model_kwargs."
            self.num_scenarios = self.decision_model_kwargs["num_scenarios"]

        # Initialize main objects
        self._initialize_decision_model()
        self._initialize_predictor()

        # Set state variables
        self._solver_calls = 0
        self._epoch_counts = 0

    # --- Abstract methods that for subclass implementation ---
    @abstractmethod
    def run_epoch(self, mode: str, epoch_num: int, metrics: Optional[list[str]] = None) -> list[dict[str, float]]:
        """
        Runs one full epoch of operations (training, validation, or testing).
        This method must be implemented by subclasses.
        It typically involves iterating over data batches, performing predictions,
        making decisions, calculating losses/metrics, and (if in 'train' mode)
        performing updates.

        Args:
            mode (str): The current mode of operation ('train', 'validation', or 'test').
            epoch_num (int): The current epoch number.
            metrics (Optional[list[str]]): The metrics that we want to record.

        Returns:
            list[dict[str, float]]: A list of dictionaries, where each dictionary contains
                scalar metrics for a batch processed during the epoch.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, data_batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Performs a single training update step for the learnable components.
        This method must be implemented by subclasses.

        Args:
            data_batch (dict[str, torch.Tensor]): A batch of training data.

        Returns:
            dict[str, float]: A dictionary of scalar loss values or metrics from this update step.
        """
        raise NotImplementedError

    # --- Core logic of the DecisionMaker ---
    def predict(self, data_batch: dict[str, torch.Tensor], output_device: Optional[str] = None) -> dict[str, torch.Tensor]:
        """
        Generates predictions for a given data batch. If a noisifier is used and the current problem mode matches
        `self.use_dist_at_mode`, it generates scenarios from the noisifier's output distribution.
        Otherwise, it uses the base predictor's forward pass.
        The raw predictions (or scenarios) are then converted into a dictionary format using `self.predictions_to_dict`.
        Finally, it incorporates any extra parameters (not predicted) from the `data_batch`
        that are required by the `decision_model`.

        Args:
            data_batch (dict[str, torch.Tensor]): Input data batch, must contain 'features'.
            output_device (Optional[str]): Device for the output
                prediction dictionary. Defaults to `self.device`.

        Returns:
            dict[str, torch.Tensor]: A dictionary of predictions, where keys are parameter
                names required by the decision model.
        """
        output_device = self.device if output_device is None else output_device
        if self.noisifier is not None and (self.problem.mode == self.use_dist_at_mode):
            distribution = self._get_noisifier_dist(data_batch)
            scenarios = self._get_scenarios(distribution)
            predictions_batch = self.predictions_to_dict(scenarios, output_device)
        else:
            features = data_batch["features"].to(torch.float32).to(self.device)
            predictions = self.predictor.forward(features)
            predictions_batch = self.predictions_to_dict(predictions, output_device)

        # Input extra parameters (that should not be predicted) into this dictionary
        predictions_batch.update({key: data_batch[key] for key in self.decision_model.extra_param_names})

        return predictions_batch

    def decide(
        self,
        predictions_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Makes decisions using the `self.decision_model` based on the provided predictions.

        Args:
            predictions_batch (dict[str, torch.Tensor]): A dictionary of predicted parameters
                (and potentially other required parameters) to be used by the decision model.

        Returns:
            dict[str, torch.Tensor]: A dictionary of decisions made by the optimization model.
        """
        batch_size = next(iter(predictions_batch.values())).shape[0]
        self._solver_calls += batch_size
        decisions_batch = self.decision_model.solve_batch(predictions_batch)
        return decisions_batch

    def predict_and_decide(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Performs a prediction step followed by a decision-making step.

        Args:
            data_batch (dict[str, torch.Tensor]): A batch of input data, typically containing
                'features' and other necessary information for prediction and decision-making.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                - prediction_batch: A dictionary of predicted parameters for the decision model.
                - decision_batch: A dictionary of decisions made by the decision model.
        """
        prediction_batch = self.predict(data_batch)
        decision_batch = self.decide(prediction_batch)
        return prediction_batch, decision_batch

    def save_best_predictor(self) -> None:
        """
        Saves a deep copy of the current `self.predictor` to `self.best_predictor`.
        This is typically called when a new best validation score is achieved.
        """
        self.best_predictor = copy.deepcopy(self.predictor)

    # --- Helper methods ---
    def _get_batch_results(self, data_batch: dict[str, torch.Tensor], metrics: Optional[list[str]] = None) -> dict[str, Any]:
        """
        Processes a single batch of data to get predictions, decisions, and evaluation metrics.
        This is a helper method typically called within `run_epoch`.

        Args:
            data_batch (dict[str, torch.Tensor]): A batch of input data.
            metrics (Optional[list[str]]): The metrics that we want to record.

        Returns:
            dict[str, Any]: A dictionary containing various results for the batch:
                - Evaluation metrics from `self.problem.evaluate()`.
                - Raw predictions (as NumPy arrays) under their parameter names.
                - Decisions (as NumPy arrays) under their variable names.
        """
        predictions_batch, decisions_batch = self.predict_and_decide(data_batch)  # Currently not logged

        batch_results = self.problem.evaluate(
            data_batch,
            decisions_batch,
            predictions_batch=predictions_batch,
            metrics=metrics,
        )
        # Note that these metrics are over all dimensions, per batch, and the epoch mean will be logged.
        for key in predictions_batch.keys():
            batch_results[key] = predictions_batch[key].cpu().detach().numpy().astype(np.float32)
        for key in decisions_batch.keys():
            batch_results[key] = decisions_batch[key].cpu().detach().numpy().astype(np.float32)

        return batch_results

    def _get_noisifier_dist(
        self,
        data_batch: dict[str, torch.Tensor],
    ) -> torch.distributions.Distribution:
        """
        Gets the output distribution from the noisifier for a given data batch.

        Args:
            data_batch (dict[str, torch.Tensor]): Input data batch, must contain 'features'.

        Returns:
            torch.distributions.Distribution: The output distribution from the noisifier.
        """
        features = data_batch["features"].to(torch.float32).to(self.device)
        distribution = self.noisifier.forward_dist(features)

        return distribution

    def _get_scenarios(self, dist: torch.distributions.Distribution) -> torch.Tensor:
        """
        Generates scenarios from a given distribution based on `self.to_decision_pars` strategy.
        Supports 'sample' (random sampling) and 'quantiles' (inverse CDF).

        Args:
            dist (torch.distributions.Distribution): The input distribution from which to generate scenarios.

        Returns:
            torch.Tensor: A tensor of scenarios. The shape is typically
                (batch_size, num_output_features, num_scenarios).
        """
        if self.to_decision_pars == "sample":
            sample = dist.sample(torch.Size([self.num_scenarios]))
            num_dims = sample.dim()
            new_order = list(range(1, num_dims)) + [0]
            scenarios = sample.permute(new_order)  # We put the scenario/sample dimension as last
        elif self.to_decision_pars == "quantiles":
            first_quantile_loc = 1 / (self.num_scenarios + 1)
            quantile_locs = torch.tensor(np.arange(first_quantile_loc, 1 - 10**-4, first_quantile_loc))
            shape = dist.batch_shape
            num_dims = len(shape)
            for _ in range(num_dims):
                quantile_locs = quantile_locs.unsqueeze(-1)
            quantile_locs = quantile_locs.expand(-1, *shape)
            quantiles = dist.icdf(quantile_locs)
            new_order = list(range(1, num_dims + 1)) + [0]
            scenarios = quantiles.permute(new_order)  # We put the scenario/sample dimension as last
        else:
            raise ValueError(f"Unsupported 'to_decision_pars' strategy: {self.to_decision_pars}")

        return scenarios

    # --- Conversion methods ---
    def predictions_to_dict(self, predictions: torch.Tensor, output_device: Optional[str] = None) -> dict[str, torch.Tensor]:
        """
        Converts a flat tensor of predictions into a dictionary.
        The dictionary maps parameter names (as defined in `self.decision_model.param_to_predict_shapes`)
        to their corresponding reshaped prediction tensors.
        If `self.to_decision_pars` indicates using scenarios (e.g., 'sample', 'quantiles') and
        the current problem mode matches `self.use_dist_at_mode` (or `use_dist_at_mode` is 'none'),
        this method first converts distributional predictions from `self.predictor` into scenarios.

        Args:
            predictions (torch.Tensor): A 2D tensor of predictions, typically of shape
                (batch_size, num_total_predicted_values). If scenarios are used, this might
                have an additional dimension for scenarios.
            output_device (Optional[str]): The device to move the output
                tensors to. Defaults to `self.device`.

        Returns:
            dict[str, torch.Tensor]: A dictionary where keys are parameter names and values
                are prediction tensors reshaped according to `self.decision_model.param_to_predict_shapes`.
                Shape of each tensor value: (batch_size, *param_shape) or
                (batch_size, *param_shape, num_scenarios) if scenarios are generated.
        """

        # Transform predictions if needed
        if (self.to_decision_pars != "none") and (self.use_dist_at_mode == "none" or self.problem.mode == self.use_dist_at_mode):
            dist = self.predictor.output_to_dist(predictions)
            predictions = self._get_scenarios(dist)

        # Transform predictions into dictionary, to be able to handle problems with multiple uncertainties
        predictions_batch = {}
        i = 0
        output_device = self.device if output_device is None else output_device
        for key, shape in self.decision_model.param_to_predict_shapes.items():
            predictions_batch[key] = predictions[:, i : i + int(np.prod(shape))].reshape(-1, *shape).to(output_device)
            i += np.prod(shape)

        return predictions_batch

    def dict_to_predictions(self, predictions_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Reverts a dictionary of named parameter predictions back into a single flat tensor.
        This is the inverse operation of `predictions_to_dict` (for the non-scenario case).

        Args:
            predictions_batch (dict[str, torch.Tensor]): A dictionary mapping parameter names
                to their prediction tensors. Tensors are expected to be of shape
                (batch_size, *param_shape) or (batch_size, *param_shape, num_scenarios).

        Returns:
            torch.Tensor: A 2D tensor of shape (batch_size, num_total_predicted_values_flat)
                or 3D if scenarios are present (batch_size, num_total_predicted_values_flat_per_scenario, num_scenarios).
                The order of concatenation follows `self.decision_model.param_to_predict_shapes`.
        """
        predictions_list = []
        for key, shape in self.decision_model.param_to_predict_shapes.items():
            predictions_list.append(predictions_batch[key].reshape(-1, int(np.prod(shape))))

        # Concatenate all tensors into a single tensor
        return torch.cat(predictions_list, dim=0)

    def dict_to_tensor(
        self,
        predictions_dict: dict[str, torch.Tensor],
        output_device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        Converts a dictionary of tensors to a single tensor by concatenating them.
        This method seems similar to `dict_to_predictions` but uses `self.problem.params_to_predict_shapes`
        as the reference for keys and order, which might differ from `self.decision_model.param_to_predict_shapes`.
        It also concatenates along `dim=1`, assuming features.

        Args:
            predictions_dict (dict[str, torch.Tensor]): Dictionary of parameter names to tensors.
                The keys should align with `self.problem.params_to_predict_shapes`.
            output_device (Optional[Union[str, torch.device]], optional): Device to move the final tensor to.
                Defaults to `self.device`.

        Returns:
            torch.Tensor: A concatenated tensor. The exact shape depends on the input tensor shapes
                and whether they have a scenario dimension.
        """
        # Set output device
        output_device = self.device if output_device is None else output_device

        # List to store tensors to concatenate
        tensors_list = []

        # Iterate through the keys in keys_to_predict
        for key in self.problem.params_to_predict_shapes:
            if key in predictions_dict:
                tensor = predictions_dict[key]
                tensor = tensor.to(output_device)  # Move to output_device if necessary
                tensors_list.append(tensor)
            else:
                raise KeyError(f"Key '{key}' not found in predictions_dict.")

        # Concatenate along the feature axis (axis 1) assuming the first axis is batch size
        # We want to keep batch size the same, so we concatenate tensors along the feature axis
        combined_tensor = torch.cat(tensors_list, dim=1)  # Concatenate along feature axis

        return combined_tensor

    # --- Initialization methods ---
    def _initialize_decision_model(self) -> None:
        """
        Initializes the decision model (`self.decision_model`).

        Based on `self.decision_model_str`, this method either creates a copy of the problem's true optimization model
        or creates a specified variant (e.g., quadratic, SAA) using methods from the problem's optimization model.
        Reference to the quadratic and SAA variants: Sufficient Decision Proxies for Decision-Focused Learning.
        Noah Schutte, Grigorii Veviurko, Krzysztof Postek, Neil Yorke-Smith. https://doi.org/10.48550/arXiv.2505.03953
        """
        if self.decision_model_str == "base":
            self.decision_model = self.problem.opt_model.create_copy()
        elif self.decision_model_str == "quadratic":
            self.decision_model = self.problem.opt_model.create_quadratic_variant()
        elif self.decision_model_str == "scenario_based":
            self.decision_model = self.problem.opt_model.create_saa_variant(**self.decision_model_kwargs)

    def _initialize_predictor(self) -> None:
        """
        Initializes the predictor model (`self.predictor`) and potentially a
        `self.noisifier` and `self.trainable_predictive_model`.

        This method configures predictor arguments (input/output dimensions, scenarios),
        handles prediction standardization by calculating means and stds from training data,
        constructs the base predictor, optionally initializes it with OLS, and sets up
        the noisifier if `self.use_noisifier` is True.
        """
        # Create a copy to not modify the original config
        predictor_kwargs_processed = self.predictor_kwargs.copy()

        # When inputs and outputs are not specified, infer them from the other specified parameters
        if "num_outputs" not in predictor_kwargs_processed:
            if self.predictor_str == "Normal":  # With distributions, we want as num_outputs the ones from the problem
                predictor_kwargs_processed["num_outputs"] = self.problem.num_predictions
            else:
                predictor_kwargs_processed["num_outputs"] = self.decision_model.num_predictions
        if "num_inputs" not in predictor_kwargs_processed:
            predictor_kwargs_processed["num_inputs"] = self.problem.num_features
        if "num_scenarios" in self.decision_model_kwargs and self.predictor_str == "MLP":
            # For now when the decision_model kwargs has num_scenarios, the predictor_kwargs gets the same number
            assert (
                "num_scenarios" in predictor_kwargs_processed and predictor_kwargs_processed["num_scenarios"] != self.decision_model_kwargs["num_scenarios"]
            ), "Num scenarios in predictor_kwargs cannot be specified and unequal to num scenarios in " "decision_model_kwargs."
            predictor_kwargs_processed["num_scenarios"] = self.decision_model_kwargs["num_scenarios"]

        # If the standardize_predictions is True or not given, then an additional layer is added to the predictor that
        # scales with the mean and he std of the training data (only implemented for MLP predictor)
        if self.init_OLS:
            assert not self.standardize_predictions, "When init_OLS is True, standardize_predictions has to be set to False"
            assert predictor_kwargs_processed.get("n_layers", 1) == 0, "When init_OLS is True, predictor_kwargs n_layers has to be set to 0 (linear model)."
        if self.standardize_predictions:
            assert (
                "shift" not in predictor_kwargs_processed and "scale" not in predictor_kwargs_processed
            ), "When standardize_predictions is not specified or True, it will overwrite scale and shift."
            predictor_kwargs_processed["shift"] = self._get_means()
            predictor_kwargs_processed["scale"] = self._get_stds()

        # Construct the base predictor
        base_predictor = IMPLEMENTED_PREDICTORS[self.predictor_str](**predictor_kwargs_processed).to(self.device)

        if self.init_OLS:
            base_predictor = self._init_OLS(base_predictor)

        # Set the predictor
        self.predictor = base_predictor

        # Initialize noisifier or use base predictor as trainable model
        if self.use_noisifier:
            self.noisifier = self._initialize_noisifier(base_predictor)
            self.trainable_predictive_model = self.noisifier
        else:
            self.noisifier = None
            self.trainable_predictive_model = self.predictor

    def _initialize_noisifier(self, base_predictor: Predictor) -> Noisifier:
        """
        Initializes the `Noisifier` component. The noisifier wraps the `base_predictor` and adds a learnable noise
        distribution to its outputs.

        Args:
            base_predictor (Predictor): The already initialized base predictor instance.

        Returns:
            Noisifier: The initialized noisifier instance.
        """
        noisifier_kwargs_processed = self.noisifier_kwargs.copy()

        if self.standardize_predictions and not noisifier_kwargs_processed.get("sigma_init", False):
            sigma_init = self._get_stds()
            noisifier_kwargs_processed["sigma_init"] = sigma_init

        noisifier = Noisifier(base_predictor=base_predictor, **noisifier_kwargs_processed).to(self.device)

        return noisifier

    def _init_OLS(self, predictor: Predictor) -> Predictor:
        """
        Initializes the weights of a linear predictor layer using Ordinary Least Squares (OLS).

        Fits a scikit-learn LinearRegression model on the training data features and target values
        (either true parameters or optimal decisions depending on `self.decision_model_str`).
        The learned coefficients and intercept are then used to set the weights and bias
        of the specified `predictor_to_init`'s linear layer.
        Handles special initialization for 'scenario_based' decision models with scenarios (using QuantileRegressor)
        and for 'Normal' predictors (initializing sigma).

        Args:
            predictor (Predictor): The predictor instance whose linear layer weights
                are to be initialized. It's assumed to have a method `get_mean_layer()`
                that returns an `nn.Linear` module.

        Returns:
            Predictor: The `predictor` with its weights initialized.
        """
        # We retrieve all the training indices
        self.problem.set_mode("train")
        idx = self.problem.generate_batch_indices(batch_size=self.problem.train_size)[0]
        train_data = self.problem.read_data(idx)
        features = train_data["features"]

        if self.decision_model_str == "quadratic":
            to_predict_list = []
            for name in self.problem.opt_model.var_names:
                to_predict_list.append(train_data[f"{name}_optimal"].reshape(self.problem.train_size, -1))
            to_predict = np.concatenate(to_predict_list, axis=1)
        else:
            to_predict_list = []
            for name in self.problem.opt_model.param_to_predict_names:
                to_predict_list.append(train_data[name].reshape(self.problem.train_size, -1))
            to_predict = np.concatenate(to_predict_list, axis=1)

        # Get OLS predictor
        skl_predictor = LinearRegression()
        skl_predictor.fit(features, to_predict)

        # Replace weights with OLS weights
        mean_layer = predictor.get_first_layer()
        if self.decision_model_str == "scenario_based" and (hasattr(self.decision_model, "num_scenarios") and self.decision_model.num_scenarios > 1):
            # We initialize with quantile regression if we have a discrete uniform predictor
            first_quantile_loc = 1 / (self.decision_model.num_scenarios + 1)
            quantile_locs = np.arange(first_quantile_loc, 1 - 10**-4, first_quantile_loc).tolist()
            weight_list = []
            bias_list = []
            for j in range(to_predict.shape[-1]):
                for _, q_loc in enumerate(quantile_locs):
                    # The predictor predicts first all scenarios for the first coefficient, so order is important
                    quantile_regressor = QuantileRegressor(quantile=q_loc, alpha=0.01)
                    quantile_regressor.fit(features, to_predict[:, j])
                    weight_list.append(quantile_regressor.coef_)
                    bias_list.append(quantile_regressor.intercept_)
            weights = np.stack(weight_list)
            biases = np.stack(bias_list)
            mean_layer.weight = nn.Parameter(torch.tensor(weights, dtype=torch.float32))
            mean_layer.bias = nn.Parameter(torch.tensor(biases, dtype=torch.float32))
        else:
            mean_layer.weight = nn.Parameter(torch.tensor(skl_predictor.coef_, dtype=torch.float32))
            mean_layer.bias = nn.Parameter(torch.tensor(skl_predictor.intercept_, dtype=torch.float32))

        if self.predictor_str == "Normal":  # We also initialize the sigma to equal the in-sample std
            predictions = skl_predictor.predict(features)
            mses = mean_squared_error(to_predict, predictions, multioutput="raw_values")
            stds = np.sqrt(mses)
            predictor.sigma_layer.bias = nn.Parameter(torch.tensor(np.log(stds), dtype=torch.float32))

        return predictor

    def _get_means(self) -> np.ndarray:
        """
        Calculates the mean of target values from the training data.

        Target values depend on `self.decision_model_str`:
        - If 'quadratic', means of optimal decisions are calculated.
        - Otherwise, means of the problem parameters to be predicted are calculated.

        Returns:
            np.ndarray: A flat NumPy array containing the concatenated means.
        """
        self.problem.set_mode("train")
        idx = self.problem.generate_batch_indices(batch_size=self.problem.train_size)[0]
        train_data = self.problem.read_data(idx)

        if self.decision_model_str == "quadratic":
            means = []
            for name in self.problem.opt_model.var_names:
                means.append(train_data[f"{name}_optimal"].mean(dim=0).detach().numpy().reshape(-1))
            means = np.concatenate(means)
        else:
            params_shapes = self.problem.params_to_predict_shapes
            params_keys = params_shapes.keys()
            means = []
            for key in params_keys:
                means.append(train_data[key].mean(dim=0).detach().numpy())
            means = np.concatenate(means)

        return means

    def _get_stds(self, min_std: float = 1e-6) -> np.ndarray:
        """
        Calculates the standard deviation of target values from the training data. Target values depend on
        `self.decision_model_str`. A minimum standard deviation is enforced to prevent issues with zero or very small
        stds. If the predictor is 'Normal' or has scenarios, stds might be repeated or augmented.

        Args:
            min_std (float, optional): Minimum value for standard deviations. Defaults to 1e-6.

        Returns:
            np.ndarray: A flat NumPy array containing the concatenated (and potentially modified) stds.
        """
        self.problem.set_mode("train")
        idx = self.problem.generate_batch_indices(batch_size=self.problem.train_size)[0]
        train_data = self.problem.read_data(idx)

        if self.decision_model_str == "quadratic":
            stds = []
            for name in self.problem.opt_model.var_names:
                stds.append(train_data[f"{name}_optimal"].std(dim=0).detach().numpy().reshape(-1))
            stds = np.concatenate(stds)
        else:
            params_shapes = self.problem.params_to_predict_shapes
            params_keys = params_shapes.keys()
            stds = []
            for key in params_keys:
                stds.append(train_data[key].std(dim=0).detach().numpy())
            stds = np.concatenate(stds)

        if self.num_scenarios is not None and self.num_scenarios > 1:
            stds = np.repeat(stds, self.num_scenarios)

        if self.predictor_str == "Normal":
            stds_std = stds.std()
            stds = np.concatenate([stds, stds_std * np.ones(stds.shape)])  # We add also and estimate of stds of the stds

        return np.maximum(stds, min_std * np.ones(stds.shape))

    def _get_quantiles(self) -> np.ndarray:
        """
        Calculates specified quantiles of target parameters from the training data.
        This method is primarily used when `self.to_decision_pars == 'quantiles' and the
        decision model is 'scenario_based', for initializing predictor biases or targets.
        It computes `self.decision_model.num_scenarios` quantiles for each parameter
        defined in `self.problem.params_to_predict_shapes`.

        Returns:
            np.ndarray: A NumPy array where rows correspond to different quantiles/scenarios
                and columns correspond to different target parameters. The exact shape depends
                on concatenation.
        """
        self.problem.set_mode("train")
        idx = self.problem.generate_batch_indices(batch_size=self.problem.train_size)[0]
        train_data = self.problem.read_data(idx)
        params_shapes = self.problem.params_to_predict_shapes
        params_keys = params_shapes.keys()

        first_quantile_loc = 1 / (self.decision_model.num_scenarios + 1)
        quantile_list = []
        for key in params_keys:
            # Calculate multiple quantiles for each variable in params_keys
            for i in range(train_data[key].shape[1]):
                quantile_locs = np.arange(first_quantile_loc, 1 - 10**-4, first_quantile_loc).tolist()
                var_quantiles = [train_data[key][:, i].quantile(q, dim=0).detach().numpy() for q in quantile_locs]
                quantile_list.append(np.stack(var_quantiles, axis=0))  # Concatenate quantiles for each variable

        quantiles = np.concatenate(quantile_list)

        return quantiles
