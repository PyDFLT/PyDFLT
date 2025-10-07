import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution

from src.predictors.multilayer_perceptron import MLPPredictor

Activation = str | nn.Module


class SampleDistribution(Distribution):
    arg_constraints = {}

    def __init__(self, average, samples):
        super().__init__()
        self.average = average
        self.samples = samples

    def sample(self, shape=None):
        if shape is None:
            shape = torch.Size()
        sample_shape = (*shape, self.average.shape[0])  # num_scenarios, batch_size
        final_shape = (*shape, *self.average.shape)
        average = self.average.detach()
        sample = average + torch.tensor(np.array(random.choices(self.samples, k=np.prod(sample_shape)))).reshape(final_shape)
        return sample


class SamplePredictor(MLPPredictor):
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_scenarios: int = 1,
        num_hidden_layers: int = 2,
        size: int = 100,
        activation: Activation = "tanh",
        output_activation: Activation = "relu",
        scale: float = 0.1,  # scale and shift are arge for the 'scale_shift' output activation function
        shift: Union[float, np.array] = 0.0,
        *args,
        **kwargs,
    ):
        MLPPredictor.__init__(
            self,
            num_inputs,
            num_outputs,
            num_scenarios,
            num_hidden_layers,
            size,
            activation,
            output_activation,
            scale,
            shift,
            *args,
            **kwargs,
        )
        # nn.Module.__init__(self,*args, **kwargs)
        self.samples = []

    def forward_dist(
        self,
        x: torch.Tensor,
    ) -> torch.distributions:
        output = self.forward(x)

        return self.output_to_dist(output)

    def to(self, device):
        return super(SamplePredictor, self).to(device)

    def output_to_dist(self, output: torch.Tensor):
        return SampleDistribution(output, self.samples)

    def update_samples(self, samples: list[torch.tensor]):
        detached_samples = [sample.detach() for sample in samples]
        self.samples = detached_samples
