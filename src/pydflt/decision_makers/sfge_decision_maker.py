from typing import Union

import numpy as np
import torch

from pydflt.decision_makers.base import DecisionMaker
from pydflt.problem import Problem


class SFGEDecisionMaker(DecisionMaker):
    """
    The SFGE decision maker is based on the paper "Silvestri, M., Berden, S., Mandi, J., Mahmutoğulları, A. İ., Amos, B.,
    Guns, T., & Lombardi, M. (2023). Score Function Gradient Estimation to Widen the Applicability of Decision-Focused
    Learning. arXiv preprint arXiv:2307.05213". The SFGE approach overcomes the zero-gradient problem in DFL by using
    a stochastic predictor at training time to smoothen the regret loss. In this codebase, we use the object Noisifier
    as the smoothing distribution around the parameterized predictive model that can be any from the list "allowed_predictors".
    Note that the Noisifier parameters are passed through "noisifier_kwargs",  which include the sigma setting.
    """

    allowed_losses: list[str] = ["objective", "regret", "relative_regret"]

    allowed_decision_models: list[str] = ["base", "quadratic", "scenario_based"]

    allowed_predictors: list[str] = [
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
        seed: Union[int, None] = None,
        predictor_kwargs: dict = None,
        noisifier_kwargs: dict = None,
        decision_model_kwargs: dict = None,
        standardize_loss: bool = True,
        num_samples: bool = 1,
    ):
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

        self.num_samples = num_samples
        self.standardize_loss = standardize_loss
        self.batch_size = batch_size
        self._set_optimizer()

    def _set_optimizer(self):
        """
        Sets the optimizer based on the trainable parameter of the predictor and the learning rate
        """
        print(f"set learning rate to {self.learning_rate}")
        self.optimizer = torch.optim.Adam(self.trainable_predictive_model.parameters(), lr=self.learning_rate)

    def update(self, data_batch: dict[str, torch.tensor], epsilon: float = 10**-5) -> dict[str, torch.tensor]:
        """
        Args:
            data_batch: contains all needed data to the update step of the predictive model

        Returns:
            the accumulated losses for the logger

        This method updates the predictive model using the MVD gradient.
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

    def run_epoch(self, mode: str, epoch_num: int, metrics: list[str] = None) -> list[dict[str, float]]:
        """
        Args:
           mode: either 'train' or 'validation' or 'test'
           epoch_num: integer counter for epoch number needed for updating of the cooling scheme

        Returns:
           epoch_results: accumulated results for logging

        This method runs an epoch based on the mode.
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
            mode_batch_results = {"%s/%s" % (mode, key): val for key, val in batch_results.items()}
            epoch_results.append(mode_batch_results)

        return epoch_results

    @staticmethod
    def standardize(input: torch.tensor, epsilon: float = 10**-5):
        """
        Args:
            input: batch losses
            epsilon: small value to avoid division by zero

        Returns:
        standardized batch losses

        Standardization is applied and serves as a baseline to reduce the variance of the SFGE, based on insights
        share by Silvestri et al. (2024)
        """
        # We standardize along the batch dimension (dim=1), using keepdim for broadcasting.
        mean_input = torch.mean(input, dim=1, keepdim=True)
        std_input = torch.std(input, dim=1, keepdim=True)

        # Broadcasting handles the element-wise operation correctly for both 1D and 2D cases.
        standardized = (input - mean_input) / (std_input + epsilon)

        return standardized
