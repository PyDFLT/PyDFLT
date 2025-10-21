from abc import abstractmethod

import numpy as np
import torch
from torch import nn


class Predictor(torch.nn.Module):
    """
    Base class for all predictors. Predictors are restricted to be instances of torch.nn.Module.

    Attributes:
        num_inputs (int): The number of inputs for the predictor.
        num_outputs (int): The total number of outputs the predictor produces.
                           If num_scenarios > 1, this is the total output (output_per_scenario * num_scenarios).
        num_scenarios (int): The number of scenarios the predictor is designed to handle.
                             If 1, the predictor outputs a single prediction.
                             If > 1, the predictor outputs multiple predictions (one per scenario).
    """

    def __init__(self, num_inputs: int, num_outputs: int, num_scenarios: int = 1):
        """
        Initializes the Predictor base class.

        Args:
            num_inputs (int): The number of inputs for the predictor.
            num_outputs (int): The total number of outputs the predictor produces.
                               If num_scenarios > 1, this is the total output (output_per_scenario * num_scenarios).
            num_scenarios (int): The number of scenarios the predictor is designed to handle.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_scenarios = num_scenarios

    def forward_mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean of the predictor's output across scenarios.

        If `num_scenarios` is 1, it directly returns the output of the `forward` method.
        If `num_scenarios` is greater than 1, it reshapes the output to separate scenarios
        and computes the mean along the scenario dimension.

        Args:
            x (torch.Tensor): The input tensor to the predictor.

        Returns:
            torch.Tensor: The mean of the predictor's output. Its shape: (batch_size, num_outputs / num_scenarios).
        """
        if self.num_scenarios == 1:
            mean = self.forward(x)
        else:
            # Get output
            output = self.forward(x)  # output shape: (batch_size, unc.par_size * num_scenarios)
            outputs_per_scenario = int(self.num_outputs / self.num_scenarios)
            shape = (outputs_per_scenario, self.num_scenarios)

            # Reshape the output to separate the scenarios
            reshaped = output.reshape(-1, *shape)  # shape: (batch_size, outputs_per_scenario, num_scenarios)
            mean = reshaped.mean(-1)  # Compute mean along the scenario dimension

        return mean

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass of the predictor. This method must be implemented by all subclasses.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of the predictor.
        """
        raise NotImplementedError

    @staticmethod
    def _set_bias(layer: nn.Module, init_bias: float | np.ndarray) -> None:
        """
        Sets the bias of a given neural network layer. This method is static and can be called directly on the class.
        It handles both scalar and array-like initial bias values.

        Args:
            layer (nn.Module): The neural network layer whose bias is to be set.
                               This layer must have a 'bias' attribute.
            init_bias (float | np.ndarray): The bias value(s). Can be a float or a numpy array.
        """
        np_init_bias = np.array(init_bias)
        if len(np_init_bias.shape) == 0:  # Scalar bias
            if init_bias != 0.0:
                with torch.no_grad():
                    layer.bias.fill_(init_bias)
        else:  # Array bias
            layer.bias = nn.Parameter(torch.tensor(np_init_bias, dtype=torch.float32))


class ScaleShift(nn.Module):
    """
    A simple scale and shift layer that can be added to a neural network's output.
    This layer performs an element-wise scaling and shifting operation on its input.
    The scale and shift parameters are not trainable (do not require gradients).

    Attributes:
        scale (torch.Tensor): A 1D tensor representing the scaling factor(s). It does not require gradients.
        shift (torch.Tensor): A 1D tensor representing the shifting factor(s). It does not require gradients.
    """

    def __init__(self, scale: float | np.ndarray = 0.1, shift: float | np.ndarray = 0):
        """
        Initializes the ScaleShift layer.

        Args:
            scale (float | np.ndarray): The scale factor(s). Can be a float or a numpy array.
            shift (float | np.ndarray): The shift factor(s). Can be a float or a numpy array.
        """
        super().__init__()
        # Ensure scale and shift are tensors and reshape them to be 1D.
        # Additionally, use buffers to ensure these values get transferred to the correct device when self.to is called.
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32, requires_grad=False).reshape(-1))
        self.register_buffer("shift", torch.tensor(shift, dtype=torch.float32, requires_grad=False).reshape(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the ScaleShift layer.

        Args:
            x (torch.Tensor): The input tensor to the layer.

        Returns:
            torch.Tensor: The scaled and shifted output tensor.
        """
        return x * self.scale + self.shift
