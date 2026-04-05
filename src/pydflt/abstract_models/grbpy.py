from abc import abstractmethod
from types import MethodType
from typing import Any

import cvxpy as cp
import gurobipy as gp
import numpy as np
import torch
from pyepo.model.opt import optModel
from tqdm import tqdm

from pydflt.abstract_models.base import MIN, OptimizationModel
from pydflt.abstract_models.cvxpy_diff import CVXPYDiffModel


class GRBPYModel(OptimizationModel):
    """
    Base class for implementing optimization models based on Gurobipy.
    This class provides the foundational structure for defining optimization problems
    that can be solved using the Gurobi solver. Subclasses should implement
    `_create_model`, `_set_params`, and `get_objective` methods to define their specific problems.

    Attributes:
        gp_model (gp.Model): The Gurobipy Model object.
        vars_dict (dict[str, gp.MVar]): Dictionary mapping decision variable names to their Gurobipy MVar objects.
        feasibility_tol (float): The feasibility tolerance for the Gurobi solver.
        rounding_decimal (int): The number of decimal places to round the decisions to.
        lazy_constraints_method (Any | None): A method for handling lazy constraints, if applicable.
    """

    gp_model: gp.Model
    vars_dict: dict[str, gp.MVar]

    def __init__(
        self,
        var_shapes: dict[str, tuple[int, ...]],
        param_to_predict_shapes: dict[str, tuple[int, ...]],
        model_sense: str,
        feasibility_tol: float = 1e-6,
        rounding_decimal: int = 4,
        extra_param_shapes: dict[str, tuple[int, ...]] | None = None,
        verbose: bool = False,
        time_limit: float | None = None,
        mip_gap: float | None = None,
    ) -> None:
        """
        Initializes the GRBPYModel.

        Args:
            var_shapes (dict[str, tuple[int, ...]]): A dictionary specifying the names and shapes of
                                                      decision variables (e.g., {'decision': (10,)}).
            param_to_predict_shapes (dict[str, tuple[int, ...]]): A dictionary specifying the names and shapes of
                                                                 parameters that must be provided prior to
                                                                 running optimization.
            model_sense (str): Specifies whether the model minimizes ('MIN') or maximizes ('MAX').
            feasibility_tol (float): The feasibility tolerance for the Gurobi solver. Defaults to 1e-6.
            rounding_decimal (int): The number of decimal places to round the decisions to. Defaults to 4.
            extra_param_shapes (dict[str, tuple[int, ...]] | None): An optional dictionary specifying additional
                                                                    parameters that change from sample to sample
                                                                    but are known.
            verbose (bool): If True, show a progress bar while solving batches. Defaults to True.
            time_limit (float | None): Optional Gurobi time limit in seconds. Defaults to None.
                                       When given, the currently best found decision is returned instead of the optimal.
                                       When no feasible decision has been found, an earlier found feasible decision
                                       is used.
            mip_gap (float | None): Optional Gurobi relative MIP gap. Defaults to None.
        """
        super().__init__(var_shapes, param_to_predict_shapes, model_sense, extra_param_shapes)
        self.gp_model, self.vars_dict = self._create_model()
        if isinstance(self, optModel):  # when the concrete model is an PyEPO optModel, we need to set attributes
            self._set_optmodel_attributes()
        self.feasibility_tol = feasibility_tol
        self.rounding_decimal = rounding_decimal
        self.verbose = verbose
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.gp_model.Params.OutputFlag = 0
        self.gp_model.setParam("FeasibilityTol", self.feasibility_tol)
        if self.time_limit is not None:
            self.gp_model.Params.TimeLimit = self.time_limit
        if self.mip_gap is not None:
            self.gp_model.Params.MIPGap = self.mip_gap
        self.lazy_constraints_method = None
        self._lazy_constraints_grb: list[gp.Constr] = []
        self.lazy_constraints: list = []
        self.supports_binding_constraints = False
        self.supports_adjacent_vertices = False
        self.gp_model.update()
        self._stored_feasible_decision: dict[str, np.ndarray] | None = None

    def relax_in_place(self) -> None:
        self.gp_model.update()
        for var in self.gp_model.getVars():
            if var.VType != gp.GRB.CONTINUOUS:
                var.VType = gp.GRB.CONTINUOUS
        self.gp_model.update()

    @abstractmethod
    def _create_model(self) -> tuple[gp.Model, dict[str, gp.MVar | gp.Var]]:
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

    @abstractmethod
    def get_objective(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Abstract method to compute the objective function value achieved by `decisions_batch`
        for the given `data_batch`. Subclasses must implement this method to define their
        specific objective function calculation.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the objective computation.
            decisions_batch (dict[str, torch.Tensor]): A dictionary containing the decision variables.
            predictions_batch (Optional[dict[str, torch.Tensor]]): An optional dictionary containing the
                                                                  predictions for relevant parameters, if applicable.
                                                                  Defaults to None.

        Returns:
            torch.Tensor: A tensor representing the objective function value(s) for the batch.
        """
        raise NotImplementedError

    def solve_batch(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Runs the optimization for a batch of data and computes the optimal decisions.
        The `data_batch` must include all parameters specified in `param_to_predict_shapes` and `extra_param_shapes`.

        Args:
            data_batch (dict[str, torch.Tensor]): A dictionary containing input data for the optimization,
                                                  including predicted parameters and extra parameters.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the computed decision variables for the batch,
                                     matching the keys in `var_shapes`.
        """
        assert all(key in data_batch for key in self.all_param_names), "data_batch must contain param_names!"

        list_decisions_batch = {key: [] for key in self.var_names}  # Here we save the output
        batch_size = len(data_batch[self.all_param_names[0]])  # Read batch size from the data
        device_data = data_batch[self.all_param_names[0]].device  # Read device on which data is stored
        # Iterate over samples
        iterator = tqdm(range(batch_size), desc="Solving", leave=False) if self.verbose else range(batch_size)
        for i in iterator:
            # We use params_list to read param values from data
            params_i = [data_batch[key][i].detach().cpu().numpy() for key in self.all_param_names]
            decisions_i = self._solve_sample(*params_i)
            # Store
            for key, val in decisions_i.items():
                list_decisions_batch[key].append(val)
        # Transform to tensor and match the device
        tensor_decisions_batch = {}
        for key in list_decisions_batch:
            tensor_decisions_batch[key] = torch.from_numpy(np.stack(list_decisions_batch[key], axis=0)).to(torch.float32).to(device_data)

        return tensor_decisions_batch

    def solve_batch_with_binding_constraints(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        """
        Solves a batch and also returns binding constraints per instance.
        Based on: https://github.com/khalil-research/CaVE
        """
        assert all(key in data_batch for key in self.all_param_names), "data_batch must contain param_names!"
        list_decisions_batch = {}
        binding_constraints = []
        batch_size = len(data_batch[self.all_param_names[0]])
        iterator = tqdm(range(batch_size), desc="Solving", leave=False) if self.verbose else range(batch_size)
        for i in iterator:
            params_i = [data_batch[key][i].detach().cpu().numpy() for key in self.all_param_names]
            decisions_i, bctr_i = self._solve_sample_with_binding_constraints(*params_i)
            binding_constraints.append(bctr_i)
            for key, val in decisions_i.items():
                if key not in list_decisions_batch:
                    list_decisions_batch[key] = []
                list_decisions_batch[key].append(val)
        decisions_batch = {key: np.stack(vals, axis=0) for key, vals in list_decisions_batch.items()}
        return decisions_batch, binding_constraints

    def solve_batch_with_adjacent_vertices(
        self,
        data_batch: dict[str, torch.Tensor],
        max_iterations_degenerate: int = 250,
        max_adjacent_vertices: int = 10000,
    ) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
        # based on: https://github.com/ML-KULeuven/Solver-Free-DFL/
        assert all(key in data_batch for key in self.all_param_names), "data_batch must contain param_names!"
        from pydflt.utils.adjacent_vertices import (
            convert_to_slack_form,
            get_adjacent_vertices,
            get_constraints_matrix_form_slack_model,
        )

        list_decisions_batch = {}
        adjacent_vertices = []
        batch_size = len(data_batch[self.all_param_names[0]])

        base_model = self.gp_model.relax() if self.gp_model.IsMIP else self.gp_model
        if getattr(base_model, "NumQConstrs", 0) > 0 or getattr(base_model, "NumQObj", 0) > 0:
            raise ValueError("Adjacent vertices require a linear model.")
        slack_model = convert_to_slack_form(base_model)
        slack_model_A, _ = get_constraints_matrix_form_slack_model(slack_model)  # noqa: N806
        slack_vars = slack_model.getVars()
        num_vars = len(base_model.getVars())

        iterator = tqdm(range(batch_size), desc="Solving", leave=False) if self.verbose else range(batch_size)
        for i in iterator:
            params_i = [data_batch[key][i].detach().cpu().numpy() for key in self.all_param_names]
            decisions_i = self._solve_sample(*params_i)
            for key, val in decisions_i.items():
                if key not in list_decisions_batch:
                    list_decisions_batch[key] = []
                list_decisions_batch[key].append(val)

            obj = gp.quicksum(params_i[0][k] * slack_vars[k] for k in range(num_vars))
            slack_model.setObjective(obj, base_model.ModelSense)
            slack_model.optimize()

            adj_vertices = get_adjacent_vertices(
                slack_model,
                slack_model_A,
                max_iterations_degenerate=max_iterations_degenerate,
                max_adjacent_vertices=max_adjacent_vertices,
            )
            adj_vertices = [np.array(av)[:num_vars] for av in adj_vertices]
            adj_vertices = list({tuple(arr.tolist()): arr for arr in adj_vertices}.values())
            if len(adj_vertices) > 1:
                sol_array = np.array(list_decisions_batch[self.var_names[0]][-1])
                adj_vertices = [v for v in adj_vertices if not np.array_equal(v, sol_array)]
            adjacent_vertices.append(np.array(adj_vertices))

        decisions_batch = {key: np.stack(vals, axis=0) for key, vals in list_decisions_batch.items()}
        return decisions_batch, adjacent_vertices

    def _solve_sample(self, *params_i_np: np.ndarray, quiet: bool = True) -> dict[str, np.ndarray]:
        """
        Solves one instance (not batched) of the optimization problem using Gurobipy.
        Note: The last line before returning the decision_dict potentially does not work when using model.AddVars()
        instead of model.AddMVar()

        Args:
            *params_i_np (np.ndarray): Parameters for the current sample.
            quiet (bool): If True, suppresses Gurobi output. Defaults to True.

        Returns:
            dict[str, np.ndarray]: A dictionary containing the computed decision variables for the sample.
        """
        if quiet:
            self.gp_model.Params.OutputFlag = 0
        self._set_params(*params_i_np)
        if self.lazy_constraints_method is None:
            self.gp_model.optimize()
        else:
            self.gp_model.optimize(self.lazy_constraints_method)
        if self.time_limit is not None and self.gp_model.Status == gp.GRB.TIME_LIMIT:
            if self.gp_model.SolCount <= 0:
                if self._stored_feasible_decision is not None:
                    print("Time limit reached without a feasible decision, stored feasible decision was used.")
                    return {key: np.array(val, copy=True) for key, val in self._stored_feasible_decision.items()}
                raise RuntimeError("Time limit reached without a feasible decision, and no stored feasible decision present.")
            else:
                print("Time limit reached, but a feasible decision was found.")

        if isinstance(next(iter(self.vars_dict.values())), list):
            # we determine the decisions differently dependent on using MVar in Gurobi or a list of variables
            decision_dict_i = {}
            for key in self.var_names:
                decision_dict_i[key] = np.array([np.round(self.vars_dict[key][i].x, self.rounding_decimal) for i in range(len(self.vars_dict[key]))])
        else:
            decision_dict_i = {key: np.round(self.vars_dict[key].x, self.rounding_decimal) for key in self.var_names}

        if self.time_limit is not None and self.gp_model.Status == gp.GRB.OPTIMAL and self._stored_feasible_decision is None:
            self._stored_feasible_decision = {key: np.array(val, copy=True) for key, val in decision_dict_i.items()}

        return decision_dict_i

    def _solve_sample_with_binding_constraints(self, *params_i_np: np.ndarray, quiet: bool = True) -> tuple[dict[str, np.ndarray], np.ndarray]:
        # based on: https://github.com/khalil-research/CaVE
        if quiet:
            self.gp_model.Params.OutputFlag = 0
        self._set_params(*params_i_np)
        existing_constrs = None
        if self.lazy_constraints_method is not None:
            self.lazy_constraints = []
            existing_constrs = self.gp_model.getConstrs()
            self.gp_model.optimize(self.lazy_constraints_method)
        else:
            self.gp_model.optimize()
        if self.time_limit is not None and self.gp_model.Status == gp.GRB.TIME_LIMIT and self.gp_model.SolCount <= 0:
            raise RuntimeError("Time limit reached without a feasible solution.")

        if existing_constrs is not None:
            self._capture_lazy_constraints(existing_constrs)

        bctr = self._get_binding_constraints()

        if isinstance(next(iter(self.vars_dict.values())), list):
            decision_dict_i = {}
            for key in self.var_names:
                decision_dict_i[key] = np.array([np.round(self.vars_dict[key][i].x, self.rounding_decimal) for i in range(len(self.vars_dict[key]))])
        else:
            decision_dict_i = {key: np.round(self.vars_dict[key].x, self.rounding_decimal) for key in self.var_names}

        if existing_constrs is not None:
            self._clear_lazy_constraints()

        return decision_dict_i, bctr

    def _get_binding_constraints(self, slack_tol: float = 1e-5) -> np.ndarray:
        # based on: https://github.com/khalil-research/CaVE
        vars_list = self.gp_model.getVars()
        num_vars = len(vars_list)
        constrs = []
        added_constrs = []
        if getattr(self, "lazy_constraints", None):
            for constr in self.lazy_constraints:
                added_constrs.append(self.gp_model.addConstr(constr))
            for var in vars_list:
                var.start = round(var.x)
            self.gp_model.update()
            self.gp_model.optimize()
        for constr in self.gp_model.getConstrs():
            if abs(constr.Slack) < slack_tol:
                coeffs = [self.gp_model.getCoeff(constr, var) for var in vars_list]
                if constr.Sense == gp.GRB.LESS_EQUAL:
                    constrs.append(coeffs)
                elif constr.Sense == gp.GRB.GREATER_EQUAL:
                    constrs.append([-coef for coef in coeffs])
                elif constr.Sense == gp.GRB.EQUAL:
                    constrs.append(coeffs)
                    constrs.append([-coef for coef in coeffs])
                else:
                    raise ValueError("Invalid constraint sense.")
        for i, var in enumerate(vars_list):
            row = [0.0] * num_vars
            if var.X <= 1e-5:
                row[i] = -1.0
                constrs.append(row)
            elif var.X >= 1 - 1e-5:
                row[i] = 1.0
                constrs.append(row)
        if added_constrs:
            self.gp_model.remove(added_constrs)
        return np.array(constrs, dtype=np.float32)

    def _clear_lazy_constraints(self) -> None:
        if not self._lazy_constraints_grb:
            return
        self.gp_model.remove(self._lazy_constraints_grb)
        self.gp_model.update()
        self._lazy_constraints_grb = []

    def _capture_lazy_constraints(self, existing_constrs: list[gp.Constr]) -> None:
        existing_set = set(existing_constrs)
        current = self.gp_model.getConstrs()
        self._lazy_constraints_grb = [constr for constr in current if constr not in existing_set]

    def create_quadratic_variant(self, cvxpy_model: bool = False) -> "OptimizationModel":
        """
        Creates a quadratic proxy (QP) variant of the problem.
        The new model will have the same constraints as the original, but its objective will be
        to minimize the squared Euclidean distance (L2 norm) between the decision variables `x`
        and predicted target vector `w` (i.e., minimize ||x-w||^2_2).

        Args:
            cvxpy_model (bool): If True, creates a CVXPYDiffModel QP variant. Otherwise, a GRBPYModel QP variant.

        Returns:
            OptimizationModel: A new instance of the model representing the QP variant.
        """
        var_shapes = self.var_shapes
        param_shapes = {}
        for key, item in var_shapes.items():
            param_key = f"{key}_par"
            param_shapes[param_key] = item

        if cvxpy_model:
            qp = CVXPYDiffModel(var_shapes, param_shapes, model_sense="MIN")
            var_domain_dict = self.get_var_domains()
            # We reset the vars with their domain
            qp.cp_vars_dict = {key: cp.Variable(shape=qp.var_shapes[key], name=key, **var_domain_dict[key]) for key in qp.var_names}
            constraints = self.get_constraints(qp.cp_vars_dict)
            obj = cp.sum(cp.sum([(qp.cp_params_dict[f"{key}_par"] - qp.cp_vars_dict[key]) ** 2 for key in qp.var_names]))
            qp.cp_model = cp.Problem(cp.Minimize(obj), constraints)
            qp.set_layer()
            qp.model_sense = MIN
        else:  # gurobipy model
            qp = self.__class__(**self.init_arguments)
            qp.model_sense = MIN
            qp._set_params = MethodType(self._set_params_qp, qp)
            qp_init_args = {
                "var_shapes": var_shapes,
                "param_to_predict_shapes": param_shapes,
                "model_sense": MIN,  # QP always minimizes
                "extra_param_shapes": None,  # No extra params for QP proxy
                "feasibility_tol": self.feasibility_tol,  # Keep original feasibility tolerance
                "rounding_decimal": self.rounding_decimal,  # Keep original rounding decimal
            }
            GRBPYModel.__init__(qp, **qp_init_args)

        return qp

    def _set_optmodel_attributes(self) -> None:
        assert len(self.vars_dict) == 1, "When the concrete model is specified as an optModel, only one decision variables can be defined"
        self.x = next(iter(self.vars_dict.values()))
        self.modelSense = self.model_sense_int
        self._model = self.gp_model

    @staticmethod
    def _set_params_qp(self: Any, *params_i: np.ndarray) -> None:
        """
        Sets the parameters for the quadratic proxy (QP) variant of the Gurobipy model.
        The objective is set to minimize the squared Euclidean distance between the decision variables
        and the provided parameters. Note that this is the default _set_params_qp, but potentially needs to be
        overwritten in the concrete_model.

        Args:
            self (Any): The instance of the GRBPYModel (or its subclass) that this method is bound to.
            *params_i (np.ndarray): The target parameters (w in ||x-w||^2_2) for the quadratic objective.
        """
        reshaped_param = []
        for param in params_i:
            reshaped_param.append(param.reshape(-1))
        c_i = np.concatenate(reshaped_param)
        if hasattr(self, "gp_model"):
            reshaped_vars = []
            for name in self.var_names:
                reshaped_vars.append(self.vars_dict[name].reshape(-1))
            second_term = gp.concatenate(reshaped_vars)
            self.gp_model.setObjective(gp.quicksum((c_i - second_term) * (c_i - second_term)))
        else:
            raise NotImplementedError("CP quadratic proxy for a Gurobipy model is not yet implemented")

    @staticmethod
    def get_var_domains() -> dict[str, dict[str, Any]] | None:
        """
        Returns a dictionary specifying the domains (e.g., bounds, types) for decision variables
        if they are to be used in a CVXPYDiffModel quadratic variant.

        Returns:
            dict[str, dict[str, Any]] | None: A dictionary where keys are variable names and values are
                                              dictionaries of CVXPY variable attributes, or None if no specific
                                              domains are defined.
        """
        raise NotImplementedError

    @staticmethod
    def get_constraints(vars_dict: dict[str, Any]) -> list[Any] | None:
        """
        Returns a list of constraints for the optimization problem.
        This method is used when creating a quadratic variant, particularly for CVXPY.

        Args:
            vars_dict (dict[str, Any]): A dictionary mapping variable names to their corresponding
                                       CVXPY Variable or Gurobipy MVar objects.

        Returns:
            list[Any] | None: A list of constraint objects (e.g., `cp.Constraint` or Gurobipy constraints),
                                 or None if no constraints are defined.
        """
        raise NotImplementedError("To create a quadratic variant, get_constraints must be implemented.")
