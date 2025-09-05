from abc import abstractmethod
from typing import Any, Optional

import numpy as np
import torch

from src.abstract_models.grbpy import GRBPYModel


class GRBPYTwoStageModel(GRBPYModel):
    """
    Base class for implementing two-stage stochastic optimization models using Gurobipy.
    This class extends GRBPYModel by introducing concepts specific to two-stage problems,
    such as fixed first-stage decisions and scenario-dependent second-stage optimization.

    Subclasses must implement:
    - `_create_model()`: To define the overall Gurobi model including first and second-stage variables.
    - `_set_params()`: To update the model's parameters for a given scenario realization.
    - `get_objective()`: To compute the objective value of the two-stage problem.
    """

    def __init__(
        self,
        variable_shapes: dict[str, tuple[int, ...]],
        param_to_predict_shapes: dict[str, tuple[int, ...]],
        model_sense: str,
        extra_param_shapes: Optional[dict[str, tuple[int, ...]]] = None,
    ) -> None:
        """
        Initializes the GRBPYTwoStageModel.

        Args:
            variable_shapes (dict[str, tuple[int, ...]]): A dictionary specifying the names and shapes of
                                                          all decision variables (including both first and second stage).
            param_to_predict_shapes (dict[str, tuple[int, ...]]): A dictionary specifying the names and shapes of
                                                                 parameters that must be provided prior to
                                                                 running optimization.
            model_sense (str): Specifies whether the model minimizes ('MIN') or maximizes ('MAX').
                               Must be either 'MIN' or 'MAX'.
            extra_param_shapes (Optional[dict[str, tuple[int, ...]]]): An optional dictionary specifying additional
                                                                       parameters that change from sample to sample
                                                                       but are known.
        """
        self.second_stage_vars_dict = {}
        GRBPYModel.__init__(
            self,
            variable_shapes,
            param_to_predict_shapes,
            model_sense,
            extra_param_shapes=extra_param_shapes,
        )

    @abstractmethod
    def _create_model(self) -> None:
        """
        Abstract method to create the Gurobipy model and initialize the `vars_dict`.
        Subclasses must implement this method to define the Gurobi problem structure,
        including variables, objective, and constraints.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_params(self, *params_i: np.ndarray) -> None:
        """
        Abstract method to set the parameters of the Gurobipy model for a single instance.
        Subclasses must implement this method to update the model's parameters based on the input.

        Args:
            *params_i (np.ndarray): Parameters (as numpy arrays) to set in the Gurobi model.
        """
        raise NotImplementedError

    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Computes the objective function value for a batch of two-stage problem instances.
        This method iterates through each sample in the batch, fixes the first-stage decisions,
        and then solves the second-stage problem to obtain the objective value for that sample.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the optimization,
                                                  including predicted parameters and extra parameters for each scenario.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing the computed first-stage decision
                                                       variables for the batch.
            predictions_batch (Optional[dict[str, torch.Tensor]]): An optional dictionary containing the
                                                                   predictions for relevant parameters, if applicable.

        Returns:
            torch.Tensor: A tensor representing the objective function value(s) for each sample in the batch.
        """
        # Read batch size from the data
        batch_size = len(data_batch[self.all_param_names[0]])

        # Iterate over samples
        objectives = []
        for i in range(batch_size):
            parameter_true_values = []
            for parameter_name in self.all_param_names:
                parameter_true_values.append(data_batch[parameter_name][i].detach().cpu().numpy())
            decisions_i_dict = {}
            for name in self.var_names:
                decisions_i_dict[name] = decisions_batch[name][i].detach().cpu().numpy()

            result = self.solve_second_stage(decisions_i_dict, *parameter_true_values)
            objective_i = result["objective_value"]
            objectives.append(objective_i)

        # Convert the list of objectives to a torch.Tensor
        objectives = torch.tensor(objectives)

        return objectives

    def solve_second_stage(self, x_assignment_dict: dict[str, np.ndarray], *parameters_i: np.ndarray) -> dict[str, Any]:
        """
        Solves the second-stage optimization problem for a single instance, given fixed
        first-stage decision variables and realized parameters.

        Args:
            x_assignment_dict (dict[str, np.ndarray]): A dictionary mapping first-stage decision variable names
                                                       to their fixed numerical assignments (as NumPy arrays).
            *parameters_i (np.ndarray): The realized parameters (as NumPy arrays) for the current scenario
                                       that affect the second-stage problem.

        Returns:
            dict[str, Any]: A dictionary containing the results of the second-stage optimization,
                            including at least the 'objective_value'.
        """
        # Set the new parameters for this realization
        self._set_params(*parameters_i)

        lb_dict = {}
        ub_dict = {}
        for var_name, x in self.vars_dict.items():
            # Save the current bounds:
            lb_dict[var_name] = [var.lb for var in x]
            ub_dict[var_name] = [var.ub for var in x]

            # Set the fixed value of first stage variable x
            # x_assign = np.array(x_assignment)
            for i, var in enumerate(x):
                var.LB = x_assignment_dict[var_name][i]
                var.UB = x_assignment_dict[var_name][i]

        # Optimize the second-stage decision variables
        self.gp_model.update()
        self.gp_model.optimize()

        # Obtain objective value
        try:
            # Try to access the objective value
            obj_value = self.gp_model.ObjVal

        except AttributeError as e:
            print(f"Error accessing objective value: {e}")
            print("No feasible solution found, set obj to 999999")
            obj_value = 999999 * self.model_sense

        # Reset lower and upper bounds
        for var_name, x in self.vars_dict.items():
            for i, var in enumerate(x):
                var.LB = lb_dict[var_name][i]
                var.UB = ub_dict[var_name][i]
        self.gp_model.update()

        # Return the objective value
        return {"objective_value": obj_value}
