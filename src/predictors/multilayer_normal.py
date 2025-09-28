from typing import Iterator, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import Parameter

from src.predictors.base import Predictor, ScaleShift

Activation = Union[str, nn.Module]


_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
    "scale_shift": ScaleShift,
}


class MLPNormalPredictor(Predictor, nn.Module):
    """
    A Multi-Layer Perceptron (MLP) predictor that outputs parameters for a Normal distribution.
    This class extends both Predictor and nn.Module to create a neural network that predicts
    both the mean and standard deviation (sigma) of a Normal distribution.

    Attributes:
        num_inputs (int): The number of input features for the predictor.
        num_outputs (int): The total number of output features (2 * original num_outputs for mean and sigma).
        num_scenarios (int): The number of scenarios the predictor is designed to handle.
        mu_layer (nn.Sequential): The neural network layers for predicting the mean.
        sigma_layer (nn.Sequential): The neural network layers for predicting the log-sigma.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_scenarios: int = 1,
        num_hidden_layers: int = 2,
        size: int = 100,
        activation: Activation = "tanh",
        output_activation: Activation = "identity",
        init_bias: Union[float, np.array] = 0.0,
        init_bias_sigma: Union[float, np.array] = 0.0,
        scale: float = 0.1,  # scale and shift are arge for the 'scale_shift' output activation function
        shift: Union[float, np.array] = 0.0,
        *args,
        **kwargs,
    ):
        """
        Initializes the MLPNormalPredictor.

        Args:
            num_inputs (int): The number of input features.
            num_outputs (int): The number of output features (will be doubled for mean and sigma).
            num_scenarios (int): The number of scenarios to predict for. Defaults to 1.
            num_hidden_layers (int): The number of hidden layers in the MLP. Defaults to 2.
            size (int): The number of neurons in each hidden layer. Defaults to 100.
            activation (Activation): The activation function for hidden layers. Defaults to 'tanh'.
            output_activation (Activation): The activation function for output layers. Defaults to 'identity'.
            init_bias (Union[float, np.array]): Initial bias for the mean layer. Defaults to 0.0.
            init_bias_sigma (Union[float, np.array]): Initial bias for the sigma layer. Defaults to 0.0.
            scale (float): Scale factor for the scale-shift output activation. Defaults to 0.1.
            shift (Union[float, np.array]): Shift factor for the scale-shift output activation. Defaults to 0.0.
            *args: Variable length argument list for nn.Module.
            **kwargs: Arbitrary keyword arguments for nn.Module.
        """
        Predictor.__init__(self, num_inputs, num_outputs, num_scenarios)
        nn.Module.__init__(self, *args, **kwargs)
        self.num_inputs = num_inputs
        self.num_outputs = 2 * num_outputs
        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str) and output_activation != "scale_shift":
            output_activation = _str_to_activation[output_activation]
        elif output_activation == "scale_shift":
            output_activation = _str_to_activation[output_activation](scale, shift)
        layers = []
        in_size = num_inputs
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(activation)
            in_size = size
        final_layer = nn.Linear(in_size, self.num_outputs)
        self._set_bias(final_layer, np.concatenate([init_bias, np.log(init_bias_sigma)]))
        layers.append(final_layer)
        layers.append(output_activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass of the MLPNormalPredictor.
        Returns concatenated mean and sigma parameters for a Normal distribution.

        Args:
            x (torch.Tensor): The input tensor to the MLP.

        Returns:
            torch.Tensor: Concatenated tensor of mean and sigma parameters.
        """
        output = self.mlp(x)
        mu, log_sigma = torch.chunk(output, 2, dim=-1)
        sigma = torch.exp(log_sigma)
        return torch.cat((mu, sigma), dim=-1)

    def forward_mean(self, x: torch.Tensor):
        """
        Performs the forward pass and returns only the mean parameters.

        Args:
            x (torch.Tensor): The input tensor to the MLP.

        Returns:
            torch.Tensor: The mean parameters only.
        """
        output = self.forward(x)
        mu, log_sigma = torch.chunk(output, 2, dim=-1)
        return mu

    def forward_dist(self, x: torch.Tensor):
        """
        Performs the forward pass and returns a Normal distribution object.

        Args:
            x (torch.Tensor): The input tensor to the MLP.

        Returns:
            torch.distributions.Normal: A Normal distribution with predicted parameters.
        """
        output = self.forward(x)
        return self.output_to_dist(output)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over the module's parameters.

        Args:
            recurse (bool): If True, yields parameters of this module and all submodules. Defaults to True.

        Returns:
            Iterator[Parameter]: An iterator over the parameters.
        """
        return self.mlp.parameters()

    @staticmethod
    def output_to_dist(output: torch.Tensor):
        """
        Converts raw output tensor to a Normal distribution.
        Static method that splits the output into mean and sigma and creates a Normal distribution.

        Args:
            output (torch.Tensor): Raw output tensor containing concatenated mean and sigma.

        Returns:
            torch.distributions.Normal: A Normal distribution with the specified parameters.
        """
        mu, sigma = torch.chunk(output, 2, dim=-1)
        dist = Normal(mu, sigma)

        return dist
