from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, LogNormal, Normal

from src.predictors import MLPNormalPredictor
from src.predictors.base import Predictor
from src.predictors.truncated_normal import TruncatedNormal


class Noisifier(nn.Module):
    """
    A class that adds noise to the output of a base predictor.

    The Noisifier can operate in several modes for determining the standard
    deviation (sigma) of the noise:
    - "fixed": Sigma is a fixed value.
    - "cooling": Sigma anneals from an initial to a final value over a set number of steps.
    - "dependent": Sigma is predicted by a linear layer based on the input.
    - "independent": Sigma is a learnable parameter, independent of the input.

    It can model the output as a Normal or LogNormal distribution.
    If the base_predictor is an MLPNormalPredictor, it can model both the
    mean and standard deviation of the underlying Normal distribution with noise.
    """

    class _DoubleDistribution(Distribution):
        """
        A distribution that combines two independent distributions.

        This class is intended for internal use within Noisifier when the base_predictor is MLPNormalPredictor,
        allowing separate noise modeling for the predicted mean and standard deviation.

        Attributes:
            arg_constraints (dict): Constraints on the distribution arguments.
            dist1 (Distribution): The first underlying distribution.
            dist2 (Distribution): The second underlying distribution.
        """

        arg_constraints = {}

        def __init__(self, dist1: Distribution, dist2: Distribution):
            """
            Initializes the DoubleDistribution.

            Args:
                dist1 (Distribution): The first distribution.
                dist2 (Distribution): The second distribution.
            """
            super().__init__()
            self.dist1 = dist1
            self.dist2 = dist2

        def sample(self, sample_shape=None) -> torch.Tensor:
            """
            Samples from both distributions and concatenates the results.

            Args:
                sample_shape (torch.Size, optional): The desired sample shape. Defaults to torch.Size().

            Returns:
                torch.Tensor: Concatenated samples from dist1 and dist2.
            """
            if sample_shape is None:
                sample_shape = torch.Size()
            sample1 = self.dist1.sample(sample_shape)
            sample2 = self.dist2.sample(sample_shape)

            return torch.cat((sample1, sample2), dim=-1)

        def log_prob(self, sample: torch.Tensor) -> torch.Tensor:
            """
            Computes the log probability of a given sample. If the sample is split, and log probabilities are computed
            for each part using the respective distributions.

            Args:
                sample (torch.Tensor): The sample for which to compute log probability.
                                     Expected to be a concatenation of samples
                                     from dist1 and dist2.

            Returns:
                torch.Tensor: Concatenated log probabilities.
            """
            sample1, sample2 = torch.chunk(sample, 2, dim=-1)  # We assume the chunks are the same size
            log_prob1 = self.dist1.log_prob(sample1)
            log_prob2 = self.dist2.log_prob(sample2)

            return torch.cat((log_prob1, log_prob2), dim=-1)

        @property
        def mean(self) -> torch.Tensor:
            """
            Returns the concatenated means of the two distributions.
            """
            return torch.cat((self.dist1.mean, self.dist2.mean), dim=-1)

        @property
        def variance(self) -> torch.Tensor:
            """
            Returns the concatenated variances of the two distributions.
            """
            return torch.cat((self.dist1.variance, self.dist2.variance), dim=-1)

    def __init__(
        self,
        base_predictor: Predictor,
        bias: bool = True,
        sigma_setting: str = "independent",
        sigma_init: Union[float, np.ndarray] = 1,
        sigma_final: float = 0.1,
        total_steps_cooling: int = 100,
        cooling_type: str = "linear",
        log_normal: bool = False,
    ):
        """
        Initializes the Noisifier.

        Args:
            base_predictor (Predictor): The underlying model that predicts the mean.
            bias (bool, optional): Whether to use a bias term in the sigma prediction layer
                                   if `sigma_setting` is "dependent". Defaults to True.
            sigma_setting (str, optional): Setting on how sigma is handled during training, options:
                                           "fixed", "dependent", "independent", "cooling".
            sigma_init (Union[float, np.ndarray], optional): Initial value for sigma.
                                                             Can be a scalar or an array matching num_outputs.
                                                             Defaults to 1.0.
            sigma_final (float, optional): Final sigma value for "cooling" mode. Defaults to 0.1.
            total_steps_cooling (int, optional): Number of steps for sigma to cool down. Defaults to 100.
            cooling_type (str, optional): Type of cooling ("linear" or "exponential"). Defaults to "linear".
            log_normal (bool, optional): If True, models the output as a LogNormal distribution.
                                         Otherwise, models as Normal (or _DoubleDistribution).
        """

        super().__init__()
        allowed_sigma_settings = ["fixed", "dependent", "independent", "cooling"]
        if sigma_setting not in allowed_sigma_settings:
            raise ValueError(f"sigma_setting must be one of {allowed_sigma_settings}. Got {sigma_setting}")
        allowed_cooling_types = ["linear", "exponential"]
        if cooling_type not in allowed_cooling_types:
            raise ValueError(f"cooling_type must be one of {allowed_cooling_types}. Got {cooling_type}")

        self.mean_layer = base_predictor
        self.num_inputs = base_predictor.num_inputs
        self.num_outputs = base_predictor.num_outputs
        self.log_normal = log_normal

        self.t = 0
        self.T = total_steps_cooling
        self.cooling_type = cooling_type

        self.sigma_setting = sigma_setting
        self.sigma_init = np.array(sigma_init)
        self.sigma_final = sigma_final
        self.sigma = None
        self._init_sigma(sigma_init, sigma_final, bias)

    def _init_sigma(self, sigma_init: Union[float, np.ndarray], sigma_final: float, bias: bool) -> None:
        """
        Initializes the sigma parameter(s) based on the sigma_setting.

        Args:
            sigma_init (Union[float, np.ndarray]): The initial value for sigma.
            sigma_final (float): The final value for sigma (used in cooling).
            bias (bool): Whether to use bias in the linear layer for "dependent" sigma.
        """
        if self.sigma_setting == "fixed" or self.sigma_setting == "cooling":
            if len(self.sigma_init.shape) == 0:
                self.sigma = torch.full((self.num_outputs,), sigma_init).to(torch.float32)
                self.sigma_init = torch.full((self.num_outputs,), sigma_init)
            else:
                self.sigma = torch.tensor(self.sigma_init, dtype=torch.float32)
                self.sigma_init = torch.tensor(self.sigma_init, dtype=torch.float32)
            self.sigma_final = torch.full((self.num_outputs,), sigma_final)
        elif self.sigma_setting == "dependent":
            self.log_sigma_layer = nn.Linear(self.num_inputs, self.num_outputs, bias)  # Predict log-sigma (log(sigma))
            if len(self.sigma_init.shape) == 0:
                with torch.no_grad():  # We set the bias to sigma_init
                    self.log_sigma_layer.bias.fill_(np.log(sigma_init))
            else:
                self.log_sigma_layer.bias = nn.Parameter(torch.tensor(np.log(self.sigma_init), dtype=torch.float32))
        elif self.sigma_setting == "independent":
            if len(self.sigma_init.shape) == 0:
                self.sigma = torch.full((self.num_outputs,), np.log(sigma_init)).to(torch.float32)
                self.sigma = nn.Parameter(self.sigma)
            else:
                self.sigma = nn.Parameter(torch.tensor(np.log(self.sigma_init), dtype=torch.float32))
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass, returning the mean prediction from the base predictor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The mean prediction from the base predictor.
        """
        mean = self.mean_layer(x)

        return mean

    def forward_dist(self, x: torch.Tensor, eps: float = 1e-7) -> Distribution:
        """
        Performs a forward pass and returns a distribution object representing the noisy output.

        Args:
            x (torch.Tensor): The input tensor.
            eps (float, optional): A small epsilon value for numerical stability, for LogNormal and TruncatedNormal.

        Returns:
            torch.distributions.Distribution: The output distribution.
        """
        # Get mu and sigma
        mean = self.mean_layer(x)
        std = self.get_sigma(x)

        if isinstance(self.mean_layer, MLPNormalPredictor):
            mu_mean, sigma_mean = torch.chunk(self.mean_layer(x), 2, dim=-1)
            mu_std, sigma_std = torch.chunk(self.get_sigma(x), 2, dim=-1)

            # Return a distribution for both the mean and the std of the Normal distribution
            dist = self._DoubleDistribution(
                Normal(mu_mean, mu_std),
                TruncatedNormal(sigma_mean + eps, sigma_std, eps, 2 * (sigma_mean.detach() + eps)),
            )
        elif self.log_normal:
            # Turning the arithmetic mean into the LogNormal mu (to keep interpretation from the layer_output)
            mu = torch.log(mean**2 / (torch.sqrt(mean**2 + std**2) + eps))
            sigma = torch.sqrt(torch.log(1 + std**2 / (mean**2 + eps))) + eps

            # Return a LogNormal distribution object with mean and std deviation
            dist = LogNormal(mu, sigma)
        else:
            # Return a Normal distribution object with mean and std deviation
            dist = Normal(mean, std)

        return dist

    def get_sigma(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the current sigma based on the setting.

        Args:
            x (torch.Tensor, optional): Input tensor, required if sigma_setting is "dependent".
                                        Defaults to None.

        Returns:
            torch.Tensor: The standard deviation tensor.
        """
        if self.sigma_setting == "dependent":
            log_sigma = self.log_sigma_layer(x)
            sigma = torch.exp(log_sigma)  # Convert log-variance to variance
        elif self.sigma_setting == "cooling":
            if self.t > self.T:
                self.t = self.T
            if self.cooling_type == "exponential":
                self.sigma = self.sigma_init * (self.sigma_final / self.sigma_init) ** (self.t / self.T)
            elif self.cooling_type == "linear":
                self.sigma = self.sigma_init + (self.sigma_final - self.sigma_init) * (self.t / self.T)
            sigma = self.sigma
        elif self.sigma_setting == "fixed":
            sigma = self.sigma
        elif self.sigma_setting == "independent":
            if isinstance(self.sigma, np.ndarray):
                self.sigma = torch.from_numpy(self.sigma)
            sigma = torch.exp(self.sigma)
        else:
            raise NotImplementedError

        if isinstance(sigma, np.ndarray):
            sigma = torch.from_numpy(self.sigma)

        return sigma

    def to(self, device: torch.device):
        """
        Moves and/or casts the parameters and buffers. This override ensures that if sigma is a tensor
        (not a parameter or part of a submodule), it's also moved to the correct device.
        """
        if self.sigma_setting in ["fixed", "cooling", "independent"]:
            self.sigma = self.sigma.to(device)
        return super(Noisifier, self).to(device)

    def update_t(self, new_t: int) -> None:
        """
        Updates the current step `t` for sigma cooling.

        Args:
            new_t (int): The new current step.
        """
        self.t = new_t

    def output_to_dist(self, output: torch.Tensor) -> Distribution:
        """
        Converts the direct output of the Noisifier (which is the mean prediction)
        into a distribution, primarily for compatibility with interfaces expecting
        a distribution from the base predictor's output format.

        Note: This method does NOT incorporate the Noisifier's own noise.
        It delegates to the base_predictor's `output_to_dist` method.
        To get the noisy distribution, use `forward_dist`.

        Args:
            output (torch.Tensor): The tensor output from the Noisifier's `forward` method
                                   (i.e., the mean prediction).

        Returns:
            torch.distributions.Distribution: A distribution object based on the
                                              base_predictor's interpretation of the output.
        """
        return self.mean_layer.output_to_dist(output)
