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
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_scenarios: int = 1,
        n_layers: int = 2,
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
        for _ in range(n_layers):
            layers.append(nn.Linear(in_size, size))
            layers.append(activation)
            in_size = size
        final_layer = nn.Linear(in_size, self.num_outputs)
        self._set_bias(final_layer, np.concatenate([init_bias, np.log(init_bias_sigma)]))
        layers.append(final_layer)
        layers.append(output_activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        output = self.mlp(x)
        mu, log_sigma = torch.chunk(output, 2, dim=-1)
        sigma = torch.exp(log_sigma)
        return torch.cat((mu, sigma), dim=-1)

    def forward_mean(self, x: torch.Tensor):
        output = self.forward(x)
        mu, log_sigma = torch.chunk(output, 2, dim=-1)
        return mu

    def forward_dist(self, x: torch.Tensor):
        output = self.forward(x)
        return self.output_to_dist(output)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.mlp.parameters()

    @staticmethod
    def output_to_dist(output: torch.Tensor):
        mu, sigma = torch.chunk(output, 2, dim=-1)
        dist = Normal(mu, sigma)

        return dist
