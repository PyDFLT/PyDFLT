import copy
import inspect
from abc import ABC, abstractmethod
from typing import Optional, Literal

import numpy as np
import torch

MAX = "MAX"
MIN = "MIN"


class OptimizationModel(ABC):
    """
    Base class that specifies the interface for all optimization models.
    To implement your own model, it is recommended to inherit from one of the children of this class,
    e.g., CVXPYModel or GRBPYModel.

    Attributes:
        var_names (list[str]): Sorted list of decision variable names.
        var_shapes (dict[str, tuple[int, ...]]): Dictionary mapping decision variable names to their shapes.
        param_to_predict_names (list[str]): Sorted list of parameter names that need to be predicted.
        param_to_predict_shapes (dict[str, tuple[int, ...]]): Dictionary mapping parameters to predict to their shapes.
        extra_param_names (list[str]): Sorted list of additional parameter names that change per sample but are known.
        extra_param_shapes (dict[str, tuple[int, ...]]): Dictionary mapping extra parameters to their shapes.
        all_param_names (list[str]): Concatenation of `param_to_predict_names` and `extra_param_names`.
        num_predictions (int): Total number of elements across all parameters to be predicted.
        num_vars (int): Total number of elements across all decision variables.
        num_params (int): Total number of elements across all parameters.
        model_sense (str): Specifies whether the model minimizes ('MIN') or maximizes ('MAX').
        init_arguments (dict[str, Any]): Stores the initial arguments used to create this model instance,
                                         used for creating variants or copies.
    """

    def __init__(
        self,
        var_shapes: dict[str, tuple[int, ...]],
        param_to_predict_shapes: dict[str, tuple[int, ...]],
        model_sense: Literal["MIN", "MAX"],
        extra_param_shapes: Optional[dict[str, tuple[int, ...]]] = None,
        num_scenarios: int = 1,
    ) -> None:
        """
        Initializes the OptimizationModel.

        Args:
            var_shapes (dict[str, tuple[int, ...]]): A dictionary specifying the names and shapes of
                                                      decision variables (e.g., {'decision': (10,)}).
            param_to_predict_shapes (dict[str, tuple[int, ...]]): A dictionary specifying the names and shapes of
                                                                 parameters that must be provided prior to
                                                                 running optimization.
            model_sense (str): Specifies whether the model minimizes ('MIN') or maximizes ('MAX').
                               Must be either 'MIN' or 'MAX'.
            extra_param_shapes (Optional[dict[str, tuple[int, ...]]]): An optional dictionary specifying additional
                                                                       parameters that change from sample to sample
                                                                       but are known.
            num_scenarios (int): The number of scenarios for multi-scenario models. Defaults to 1.
        """

        assert model_sense.upper() in [
            MIN,
            MAX,
        ], f"model_sense must be {MIN} for minimization or {MAX} for maximization!"

        # We store the init_arguments from the child class, so we can create model variants with the same parameters
        self.init_arguments = {
            name: getattr(self, name) for name in inspect.signature(type(self).__init__).parameters if name != "self" and hasattr(self, name)
        }

        # Parse and save input arguments
        self.var_names = sorted(var_shapes.keys())
        self.var_shapes = var_shapes

        self.param_to_predict_names = sorted(param_to_predict_shapes.keys())
        self.param_to_predict_shapes = param_to_predict_shapes
        extra_param_shapes = extra_param_shapes or {}
        self.extra_param_names = sorted(extra_param_shapes.keys())
        self.extra_param_shapes = extra_param_shapes
        self.all_param_names = self.param_to_predict_names + self.extra_param_names

        self.num_predictions = np.sum([np.prod(self.param_to_predict_shapes[name]) for name in self.param_to_predict_names])
        self.num_vars = sum([np.prod(shape) for key, shape in var_shapes.items()])
        self.num_params = sum([np.prod(shape) for shape in self.param_to_predict_shapes.values()])

        self.model_sense = model_sense

    @property
    def model_sense_int(self) -> int:
        """
        Returns the model sense as an integer: 1 for minimization ('MIN') and -1 for maximization ('MAX').

        Returns:
            int: 1 if model_sense is 'MIN', -1 if model_sense is 'MAX'.
        """
        if self.model_sense == MIN:
            return 1
        elif self.model_sense == MAX:
            return -1
        else:
            raise NotImplementedError

    @abstractmethod
    def solve_batch(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the optimization for a batch of data and computes the optimal decisions.
        The `data_batch` must include all parameters specified in `param_to_predict_shapes`
        and `extra_param_shapes`.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the optimization,
                                                  including predicted parameters and extra parameters.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed decision variables for the batch,
                                     matching the keys in `var_shapes`.
        """
        raise NotImplementedError("Subclasses must implement solve_batch method.")

    @abstractmethod
    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Computes the objective function value achieved by `decisions_batch` for the given `data_batch`.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the optimization.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing the decision variables.
            predictions_batch (Optional[dict[str, torch.Tensor]]): An optional dictionary containing the
                                                                  predictions for relevant parameters, if applicable.
                                                                  Defaults to None.

        Returns:
            torch.Tensor: A tensor representing the objective function value(s) for the batch.
        """
        raise NotImplementedError("Subclasses must implement get_objective method.")

    def get_penalty(self, x_u: torch.Tensor, data_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the constraint violation (penalty) for a given point `x_u`.
        This method is used internally by some algorithms for constraint handling.

        Args:
            x_u (torch.Tensor): The point (decision variables) for which to compute the penalty.
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data.

        Returns:
            torch.Tensor: A tensor representing the penalty value(s).
        """
        raise NotImplementedError

    def get_reward_gradient(
        self,
        decisions_batch: dict[str, torch.Tensor],
        data_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the gradient of the reward (objective) function with respect to the solution.

        Args:
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing the decision variables.
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data.

        Returns:
            torch.Tensor: A tensor representing the gradient of the reward function.
        """
        raise NotImplementedError

    def create_quadratic_variant(self) -> "OptimizationModel":
        """
        Creates a quadratic proxy (QP) variant of the problem.
        The new model will have the same constraints as the original, but its objective will be
        to minimize the squared Euclidean distance (L2 norm) between the decision variables `x`
        and predicted target vector `w` (i.e., minimize ||x-w||^2_2).

        Returns:
            OptimizationModel: A new instance of the model representing the QP variant.
        """
        raise NotImplementedError

    def evaluate(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: Optional[dict[str, torch.Tensor]] = None,
        epsilon: float = 1e-5,
        metrics: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        """
        Evaluates a batch of decisions by computing various metrics such as objective value,
        absolute regret, relative regret, and symmetric relative regret.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the evaluation.
                                                  Expected to include 'objective_optimal' if regret is to be computed.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing the computed decision variables.
            predictions_batch (Optional[dict[str, torch.Tensor]]): An optional dictionary containing the
                                                                  predictions for relevant parameters. Defaults to None.
            epsilon (float): A small value added to the denominator in relative regret calculations to prevent
                             division by zero. Defaults to 1e-5.
            metrics (Optional[list[str]]): List of metrics on which to evaluate.

        Returns:
            dict[str, np.ndarray]: A dictionary containing evaluation metrics, including:
                                   - 'objective': The objective function value(s).
                                   - 'abs_regret' (optional): Absolute regret if 'objective_optimal' is in `data_batch`.
                                   - 'rel_regret' (optional): Relative regret if 'objective_optimal' is in `data_batch`.
                                   - 'sym_rel_regret' (optional): Symmetric relative regret if 'objective_optimal'
                                                                  is in `data_batch`.
        """
        if metrics is None:  # Check if metrics was not provided
            metrics = ["abs_regret"]

        eval_dict = {}
        if "objective" in metrics or "abs_regret" in metrics or "rel_regret" in metrics or "sym_rel_regret" in metrics:
            objectives = self.get_objective(data_batch, decisions_batch, predictions_batch)
            if "objective" in metrics:
                eval_dict["objective"] = objectives.cpu().cpu().detach().numpy().astype(np.float32)
            if "abs_regret" in metrics or "rel_regret" in metrics or "sym_rel_regret" in metrics:
                optimal_objectives = data_batch["objective_optimal"]
                regret = (objectives - optimal_objectives) * float(self.model_sense_int)
                if "abs_regret" in metrics:
                    eval_dict["abs_regret"] = regret.cpu().detach().numpy().astype(np.float32)
                if "rel_regret" in metrics or "sym_rel_regret" in metrics:
                    relative_regret = regret / optimal_objectives
                    if "rel_regret" in metrics:
                        eval_dict["rel_regret"] = relative_regret.cpu().detach().numpy().astype(np.float32)
                    if "sym_rel_regret" in metrics:
                        symmetric_relative_regret = regret / (abs(optimal_objectives) + abs(objectives) + epsilon)
                        eval_dict["sym_rel_regret"] = symmetric_relative_regret.cpu().detach().numpy().astype(np.float32)

        return eval_dict

    def create_saa_variant(self, num_scenarios: int) -> "OptimizationModel":
        """
        Creates a Sample Average Approximation (SAA) variant of the optimization problem.
        This version incorporates multiple scenarios for stochastic optimization.

        Args:
            num_scenarios (int): The number of scenarios to include in the SAA variant.

        Returns:
            OptimizationModel: A new instance of the model representing the SAA variant.
        """
        assert "num_scenarios" in self.init_arguments, "Concrete model needs to have a num_scenarios argument if you want to create a SAA variant."
        assert all(
            arg in self.init_arguments for arg in inspect.signature(type(self).__init__).parameters if arg != "self"
        ), "Concrete model needs to have all attributes set to arguments if you want to create a SAA variant."
        init_arguments = {key: item for key, item in self.init_arguments.items() if key != "num_scenarios"}

        return self.__class__(**init_arguments, num_scenarios=num_scenarios)

    def create_copy(self) -> "OptimizationModel":
        """
        Creates a shallow copy of the current model instance.

        Returns:
            OptimizationModel: A new instance of the same model with the same attributes.
        """
        return copy.copy(self)
