from typing import ClassVar

import numpy as np
import torch

from pydflt.decision_makers.base import DecisionMaker
from pydflt.problem import Problem


class SFGEDecisionMaker(DecisionMaker):
    """
    The SFGE decision maker is based on the paper "Silvestri, M., Berden, S., Mandi, J., Mahmutogullari, A. I., Amos, B.,
    Guns, T., & Lombardi, M. (2023). Score Function Gradient Estimation to Widen the Applicability of Decision-Focused
    Learning. arXiv preprint arXiv:2307.05213". The SFGE approach overcomes the zero-gradient problem in DFL by using
    a stochastic predictor at training time to smoothen the regret loss. In this codebase, we use the object Noisifier
    as the smoothing distribution around the parameterized predictive model that can be any from the list "allowed_predictors".
    Note that the Noisifier parameters are passed through "noisifier_kwargs",  which include the sigma setting.
    """

    allowed_losses: ClassVar[list[str]] = ["objective", "regret", "relative_regret"]

    allowed_decision_models: ClassVar[list[str]] = ["base", "quadratic", "scenario_based"]

    allowed_predictors: ClassVar[list[str]] = [
        "Normal",
        "DiscreteUniform",
        "MLP",
    ]

    def __init__(
        self,
        problem: Problem,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device_str: str = "cpu",
        predictor_str: str = "MLP",
        decision_model_str: str = "base",
        loss_function_str: str = "regret",
        to_decision_pars: str = "none",
        use_dist_at_mode: str = "none",
        standardize_predictions: bool = True,
        init_OLS: bool = False,
        seed: int | None = None,
        predictor_kwargs: dict | None = None,
        noisifier_kwargs: dict | None = None,
        decision_model_kwargs: dict | None = None,
        standardize_loss: bool = True,
        num_samples: int = 1,
    ) -> None:
        super().__init__(
            problem,
            learning_rate,
            device_str,
            predictor_str,
            decision_model_str,
            loss_function_str,
            to_decision_pars,
            use_dist_at_mode,
            True,
            standardize_predictions,
            init_OLS,
            seed,
            predictor_kwargs,
            noisifier_kwargs,
            decision_model_kwargs,
        )

        """
        Initializes the SFGEDecisionMaker.

        Args:
            problem (Problem): The problem instance containing data and optimization model.
            learning_rate (float): Learning rate for the Adam optimizer. Defaults to 1e-4.
            batch_size (int): Number of samples per training batch. Defaults to 32.
            device_str (str): Device for computations ('cpu' or 'cuda'). Defaults to 'cpu'.
            predictor_str (str): Type of predictor to use (must be in `allowed_predictors`). Defaults to 'MLP'.
            decision_model_str (str): Type of decision model. Defaults to 'base'.
            loss_function_str (str): Loss function to use (must be in `allowed_losses`). Defaults to 'regret'.
            to_decision_pars (str): Strategy for converting predictions to decision parameters. Defaults to 'none'.
            use_dist_at_mode (str): Mode in which to use the distributional predictor. Defaults to 'none'.
            standardize_predictions (bool): Whether to standardize predictions before passing to the decision model.
                Defaults to True.
            init_OLS (bool): Whether to initialize the predictor with OLS weights. Defaults to False.
            seed (int | None): Random seed for reproducibility. Defaults to None.
            predictor_kwargs (dict | None): Additional keyword arguments for predictor initialization. Defaults to None.
            noisifier_kwargs (dict | None): Keyword arguments for the Noisifier (e.g. sigma). Defaults to None.
            decision_model_kwargs (dict | None): Additional keyword arguments for decision model initialization.
                Defaults to None.
            standardize_loss (bool): Whether to standardize the SFGE loss across the batch. Defaults to True.
            num_samples (int): Number of samples to draw from the distributional predictor per instance.
                Defaults to 1.
        """
        self.num_samples = num_samples
        self.standardize_loss = standardize_loss
        self.batch_size = batch_size
        self._set_optimizer()

    def _set_optimizer(self) -> None:
        """
        Sets the optimizer based on the trainable parameter of the predictor and the learning rate
        """
        print(f"set learning rate to {self.learning_rate}")
        self.optimizer = torch.optim.Adam(self.trainable_predictive_model.parameters(), lr=self.learning_rate)

    def update(self, data_batch: dict[str, torch.Tensor], epsilon: float = 10**-5) -> dict[str, torch.Tensor]:
        """
        Updates the predictive model using the MVD (Measure-Valued Derivative) gradient.

        Samples from the distributional predictor, evaluates the objective for each sample,
        computes the SFGE loss via the log-derivative trick, and performs one gradient step.

        Args:
            data_batch (dict[str, torch.Tensor]): All data needed for the update step,
                including features, true parameters, and optimal objectives.
            epsilon (float): Unused parameter kept for API compatibility. Defaults to 1e-5.

        Returns:
            dict[str, torch.Tensor]: Accumulated losses and diagnostics for the logger,
                containing keys 'loss', 'eval', 'solver_calls', and 'sigma'.
        """

        # Obtain the distributional predictor and sample
        distribution = self._get_noisifier_dist(data_batch)
        samples = distribution.sample((self.num_samples,))

        # Get log probabilities
        individual_log_probs = distribution.log_prob(samples)
        log_probs = individual_log_probs.sum(dim=-1)  # sum over num_parameters dimension (the last one)

        # Get objective value per sample
        objectives = torch.zeros(samples.shape[:2])  # per sample, per batch
        for i in range(self.num_samples):
            # Put samples in prediction batch to get decisions and objective values
            predictions_batch = self.predictions_to_dict(samples[i])
            decisions_batch = self.decide(predictions_batch)
            sample_objectives = self.problem.opt_model.get_objective(data_batch, decisions_batch, predictions_batch=predictions_batch)
            objectives[i] = sample_objectives

        # Compute loss function value
        optimal_objectives = data_batch["objective_optimal"]
        if self.loss_function_str == "regret":
            loss_terms = (objectives - optimal_objectives) * self.problem.opt_model.model_sense_int
        elif self.loss_function_str == "objective":
            loss_terms = objectives * self.problem.opt_model.model_sense_int
        elif self.loss_function_str == "relative_regret":
            loss_terms = (objectives - optimal_objectives) / optimal_objectives * self.problem.opt_model.model_sense_int

        # loss_terms = loss_terms.mean(dim=0)  # take mean over samples
        # base_loss = loss_terms.detach().numpy().astype(np.float32)
        # loss_terms = loss_terms.float()

        if self.standardize_loss:
            loss_terms = self.standardize(loss_terms)

        # Compute surrogate loss for gradient
        base_loss = loss_terms.mean(dim=0).detach().numpy().astype(np.float32)
        loss = (loss_terms * log_probs).mean(dim=0)
        logger_loss = loss.detach().numpy().astype(np.float32)
        loss_mean = torch.mean(loss)

        # Update
        self.optimizer.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        # Logging
        log_dict = {
            "loss": logger_loss,
            "eval": base_loss,
            "solver_calls": self._solver_calls,
            "sigma": torch.sqrt(distribution.variance).detach().numpy().astype(np.float32),
        }
        return log_dict

    def run_epoch(self, mode: str, epoch_num: int, metrics: list[str] | None = None) -> list[dict[str, float]]:
        """
        Runs one complete epoch in the specified mode (train/validation/test).

        In train mode, updates the noisifier's cooling schedule if configured, then processes
        each batch by calling `update`. In validation/test mode, evaluates the predictor using
        `_get_batch_results`.

        Args:
            mode (str): The mode to run ('train', 'validation', or 'test').
            epoch_num (int): The current epoch number, used to update the noisifier cooling scheme.
            metrics (list[str] | None): List of additional metrics to evaluate. Defaults to None.

        Returns:
            list[dict[str, float]]: List of result dictionaries, one per batch processed.
        """
        assert mode in [
            "train",
            "validation",
            "test",
        ], "Mode must be train/validation/test!"

        # Switch predictor and problem mode to train/evaluation
        self.trainable_predictive_model.train() if mode == "train" else self.trainable_predictive_model.eval()
        self.problem.set_mode(mode)

        # Update dist predictor t
        if self.noisifier.sigma_setting == "cooling":
            self.noisifier.update_t(epoch_num)

        # Initialize dictionary with the results
        epoch_results = []

        # Run
        for idx in self.problem.generate_batch_indices(self.batch_size):
            data_batch = self.problem.read_data(idx)
            if mode == "train":
                batch_results = self.update(data_batch)
            else:
                batch_results = self._get_batch_results(data_batch, metrics=metrics)
            mode_batch_results = {f"{mode}/{key}": val for key, val in batch_results.items()}
            mode_batch_results["batch_size"] = len(idx)
            epoch_results.append(mode_batch_results)

        return epoch_results

    @staticmethod
    def standardize(input: torch.Tensor, epsilon: float = 10**-5):
        """
        Standardizes a batch of losses along the batch dimension.

        Standardization serves as a variance-reduction baseline for the SFGE gradient estimator,
        based on insights from Silvestri et al. (2024).

        Args:
            input (torch.Tensor): Batch losses to standardize.
            epsilon (float): Small value added to the standard deviation to avoid division by zero.
                Defaults to 1e-5.

        Returns:
            torch.Tensor: Standardized batch losses with zero mean and unit variance along dim 1.
        """
        # We standardize along the batch dimension (dim=1), using keepdim for broadcasting.
        mean_input = torch.mean(input, dim=1, keepdim=True)
        std_input = torch.std(input, dim=1, keepdim=True)

        # Broadcasting handles the element-wise operation correctly for both 1D and 2D cases.
        standardized = (input - mean_input) / (std_input + epsilon)

        return standardized
