# Partially based on: https://github.com/facebookresearch/LANCER

from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from pydflt.predictors.base import Predictor, ScaleShift

Activation = str | nn.Module


_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}


class MLPPredictor(Predictor, nn.Module):
    """
    A Multi-Layer Perceptron (MLP) predictor that extends both Predictor and nn.Module.
    This class constructs a feed-forward neural network with configurable layers,
    activation functions, and an optional scale-shift layer at the output.

    Attributes:
        mlp (nn.Sequential): The sequential neural network comprising the MLP layers.
        num_inputs (int): The number of input features for the predictor.
        num_outputs (int): The total number of output features the predictor produces.
        num_scenarios (int): The number of scenarios the predictor is designed to handle.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_scenarios: int = 1,
        num_hidden_layers: int = 2,  # 0 for a linear predictor
        size: int = 252,
        activation: Activation = "leaky_relu",
        output_activation: Activation = "identity",
        scale: float | np.ndarray = 1.0,
        shift: float | np.ndarray = 0.0,
        *args,
        **kwargs,
    ):
        """
        Initializes the MLPPredictor.

        Args:
            num_inputs (int): The number of input features.
            num_outputs (int): The number of output features.
            num_scenarios (int): The number of scenarios to predict for. Defaults to 1.
            num_hidden_layers (int): The number of hidden layers in the MLP. 0 means a linear predictor. Defaults to 2.
            size (int): The number of neurons in each hidden layer. Defaults to 252.
            activation (Activation): The activation function to use for hidden layers.
                                     Can be a string (e.g., 'relu', 'leaky_relu') or an nn.Module.
                                     Defaults to 'leaky_relu'.
            output_activation (Activation): The activation function to use for the output layer.
                                            Can be a string or an nn.Module. Defaults to 'identity'.
            scale (float | np.ndarray): The scale factor for the output scale-shift layer. Defaults to 1.0.
            shift (float | np.ndarray): The shift factor for the output scale-shift layer. Defaults to 0.0.
            *args: Variable length argument list to be passed to nn.Module.__init__.
            **kwargs: Arbitrary keyword arguments to be passed to nn.Module.__init__.
        """

        Predictor.__init__(self, num_inputs, num_outputs, num_scenarios)
        nn.Module.__init__(self, *args, **kwargs)

        if isinstance(activation, str):
            activation = _str_to_activation[activation]
        if isinstance(output_activation, str):
            output_activation = _str_to_activation[output_activation]

        layers = []
        in_size = num_inputs
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(activation)
            in_size = size

        # Apply output layer and output activation
        layers.append(nn.Linear(in_size, num_outputs))
        layers.append(output_activation)

        # Scale shift is always applied. Default doesn't do anything
        scale_shift_layer = ScaleShift(scale, shift)
        layers.append(scale_shift_layer)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the MLPPredictor.

        Args:
            x (torch.Tensor): The input tensor to the MLP.

        Returns:
            torch.Tensor: The output tensor from the MLP.
        """
        output = self.mlp(x)
        return output

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Returns an iterator over the module's parameters.

        Args:
            recurse (bool): If True, then yields parameters of this module
                            and all submodules. Otherwise, yields only parameters
                            that are direct members of this module. Defaults to True.

        Returns:
            Iterator[Parameter]: An iterator over the parameters.
        """
        return self.mlp.parameters(recurse=recurse)

    def get_first_layer(self) -> nn.Module:
        """
        Returns the first layer of the MLP. This can be used to adjust the weights of this layer, which is relevant
        when we are working with linear predictors for example (then it is the only layer).

        Returns:
            nn.Module: The first layer of the MLP.
        """
        return self.mlp[0]
