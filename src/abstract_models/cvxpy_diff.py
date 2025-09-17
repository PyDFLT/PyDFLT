from abc import abstractmethod
from typing import Optional

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

from src.abstract_models.base import OptimizationModel


class CVXPYDiffModel(OptimizationModel):
    """
    Base class for implementing differentiable optimization models based on CVXPY Layers.
    To implement your own problem, edit the '_create_cp_model' and 'get_objective' methods.

    Attributes:
        layer (CvxpyLayer): The CVXPY Layer object that represents the differentiable optimization problem.
        cp_model (cp.Problem): The CVXPY Problem object.
        cp_vars_dict (dict[str, cp.Variable]): Dictionary mapping decision variable names to their CVXPY Variable objects.
        cp_params_dict (dict[str, cp.Parameter]): Dictionary mapping parameter names to their CVXPY Parameter objects.
    """

    layer: CvxpyLayer
    cp_model: cp.Problem
    cp_vars_dict: dict[str, cp.Variable]
    cp_params_dict: dict[str, cp.Parameter]

    def __init__(
        self,
        var_shapes: dict[str, tuple[int, ...]],
        param_to_predict_shapes: dict[str, tuple[int, ...]],
        model_sense: str,
        extra_param_shapes: Optional[dict[str, tuple[int, ...]]] = None,
    ) -> None:
        """
        Initializes the CVXPYDiffModel.

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
        """
        super().__init__(var_shapes, param_to_predict_shapes, model_sense, extra_param_shapes)
        self._init_vars_and_params()
        self._create_cp_model()
        self.set_layer()

    @abstractmethod
    def _create_cp_model(self) -> None:
        """
        Creates the CVXPY optimization model (`self.cp_model`).
        Users should override this method to implement their specific optimization problem.
        """
        raise NotImplementedError

    @abstractmethod
    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Computes the objective function value achieved by `decisions_batch` for the given `data_batch`.
        Users should override this method to define their specific objective function.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the objective computation.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing the decision variables.
            predictions_batch (Optional[dict[str, torch.Tensor]]): An optional dictionary containing the
                                                                  predictions for relevant parameters, if applicable.

        Returns:
            torch.Tensor: A tensor representing the objective function value(s) for the batch.
        """
        raise NotImplementedError

    def _init_vars_and_params(self) -> None:
        """
        Initializes CVXPY objects needed to build the CVXPY optimization problem.
        This includes creating CVXPY Variable objects for decision variables and CVXPY Parameter objects for parameters.
        """
        self.cp_vars_dict = {key: cp.Variable(shape=self.var_shapes[key], name=key) for key in self.var_names}
        self.cp_params_dict = {key: cp.Parameter(shape=self.param_to_predict_shapes[key], name=key) for key in self.param_to_predict_names}
        self.cp_params_dict.update({key: cp.Parameter(shape=self.extra_param_shapes[key], name=key) for key in self.extra_param_names})

    def set_layer(self) -> None:
        """
        Creates a CVXPY Layer based on the CVXPY optimization problem (`self.cp_model`).
        This method is called after the `_create_cp_model` method to set up the differentiable layer.
        """
        if hasattr(self, "cp_model") and isinstance(self.cp_model, cp.Problem):
            self.layer = CvxpyLayer(
                problem=self.cp_model,
                variables=[self.cp_vars_dict[key] for key in self.var_names],
                parameters=[self.cp_params_dict[key] for key in self.all_param_names],
            )

    def solve_batch(
        self,
        data_batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Runs the optimization for a batch of data and computes the optimal decisions.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the optimization,
                                                  including predicted parameters and extra parameters.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed decision variables for the batch,
                                     matching the keys in `var_shapes`.
        """
        assert all(key in data_batch.keys() for key in self.all_param_names), "data_batch must contain all param_names!"

        layer_input = [data_batch[key] for key in self.all_param_names]
        layer_output = self.layer(*layer_input, solver_args={"eps_abs": 10**-9, "eps_rel": 10**-9})
        decision_batch = dict(zip(self.var_names, layer_output, strict=False))

        return decision_batch

    def create_quadratic_variant(self) -> "OptimizationModel":
        """
        Creates a quadratic proxy (QP) variant of the problem.
        The new model will have the same constraints as the original, but its objective will be
        to minimize the squared Euclidean distance (L2 norm) between the decision variables `x`
        and predicted target vector `w` (i.e., minimize ||x-w||^2_2).

        Returns:
            OptimizationModel: A new instance of the model representing the QP variant.
        """
        qp_model = self.__class__(**self.init_arguments)
        var_shapes_qp = self.var_shapes  # Variables are the same
        param_shapes_qp = {}  # In QP proxy we have one parameter per decision variable
        for key, shape in var_shapes_qp.items():
            param_shapes_qp[f"{key}_par"] = shape

        # We initialize the parent class to have different var_shapes, param_shapes and params_to_predict
        OptimizationModel.__init__(qp_model, var_shapes_qp, param_shapes_qp, model_sense="MIN")
        qp_model._init_vars_and_params()

        # Now we recreate the model, and set the layer. We use the same vars and constraints as the original model!
        qp_model.cp_vars_dict = self.cp_vars_dict
        obj = cp.sum([cp.sum_squares(qp_model.cp_params_dict[f"{key}_par"] - qp_model.cp_vars_dict[key]) for key in qp_model.var_names])
        qp_model.cp_model = cp.Problem(cp.Minimize(obj), self.cp_model.constraints)
        qp_model.set_layer()

        return qp_model
