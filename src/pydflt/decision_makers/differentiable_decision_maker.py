from typing import Any, ClassVar

import numpy as np
import torch
from pyepo.func import (
    SPOPlus,
    blackboxOpt,
    implicitMLE,
    negativeIdentity,
    perturbedFenchelYoung,
    perturbedOpt,
)
from pyepo.func.abcmodule import optModule
from pyepo.func.utlis import _cache_in_pass, _solve_in_pass, _solve_or_cache

from pydflt.decision_makers.base import DecisionMaker
from pydflt.problem import Problem
from pydflt.utils.cave import InnerConeAlignedCosine
from pydflt.utils.lava import LAVA

PYEPO_LOSS_FUNCTIONS = {
    "SPOPlus": SPOPlus,
    "perturbedOpt": perturbedOpt,
    "perturbedFenchelYoung": perturbedFenchelYoung,
    "implicitMLE": implicitMLE,
    "blackboxOpt": blackboxOpt,
    "negativeIdentity": negativeIdentity,
}
"""
Mapping of PyEPO loss function names to their corresponding implementations.
These are differentiable loss functions designed for end-to-end learning in
optimization problems.
"""


class DifferentiableDecisionMaker(DecisionMaker):
    """
    A decision maker that supports differentiable optimization through PyEPO loss functions.
    This class extends the base DecisionMaker to provide training capabilities using
    gradient-based optimization with various differentiable loss functions.

    Supports multiple loss functions including:
    - Traditional losses: MSE, objective, regret
    - PyEPO differentiable losses: SPOPlus, perturbedOpt, perturbedFenchelYoung, etc.
    - Smooth loss for quadratic decision models

    Attributes:
        batch_size (int): The batch size used for training iterations.
        residual_SAA (bool): Whether to use residual Sample Average Approximation.
        residual_SAA_scenarios (int): Number of scenarios for residual SAA.
        optimizer (torch.optim.Optimizer): The optimizer used for training the predictive model.
        loss_function (callable): The loss function used for training.
    """

    allowed_losses: ClassVar[list[str]] = [
        "mse",
        "objective",
        "regret",
        "cave",
        "lava",
        "SPOPlus",
        "perturbedOpt",
        "perturbedFenchelYoung",
        "implicitMLE",
        "blackboxOpt",
        "negativeIdentity",
        "smooth",
    ]

    allowed_decision_models: ClassVar[list[str]] = ["base", "quadratic", "scenario_based"]

    allowed_predictors: ClassVar[list[str]] = [
        "LinearSKL",
        "MLP",
        "DiscreteUniform",
        "BoundedLinear",
        "Normal",
        "Sample",
    ]

    def __init__(
        self,
        problem: Problem,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device_str: str = "cpu",
        predictor_str: str = "MLP",
        decision_model_str: str = "base",
        loss_function_str: str = "objective",
        to_decision_pars: str = "none",
        use_dist_at_mode: str = "none",
        standardize_predictions: bool = True,
        init_OLS: bool = False,
        seed: int | None = None,
        predictor_kwargs: dict | None = None,
        decision_model_kwargs: dict | None = None,
        residual_SAA: bool = False,
        residual_SAA_scenarios: int = 1,
    ) -> None:
        """
        Initializes the DifferentiableDecisionMaker.

        Args:
            problem (Problem): The problem instance containing data and optimization model.
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-4.
            batch_size (int): Batch size for training iterations. Defaults to 32.
            device_str (str): Device for computations ('cpu' or 'cuda'). Defaults to 'cpu'.
            predictor_str (str): Type of predictor to use. Defaults to 'MLP'.
            decision_model_str (str): Type of decision model ('base', 'quadratic', 'scenario_based'). Defaults to 'base'.
            loss_function_str (str): Loss function to use for training. Defaults to 'objective'.
            to_decision_pars (str): Strategy for converting predictions to decision parameters. Defaults to 'none'.
            use_dist_at_mode (str): When to use distributional predictions. Defaults to 'none'.
            standardize_predictions (bool): Whether to standardize predictions. Defaults to True.
            init_OLS (bool): Whether to initialize with OLS. Defaults to False.
            seed (int | None): Random seed for reproducibility. Defaults to None.
            predictor_kwargs (dict | None): Additional arguments for predictor initialization. Defaults to None.
            decision_model_kwargs (dict | None): Additional arguments for decision model initialization. Defaults to None.
            residual_SAA (bool): Whether to use residual Sample Average Approximation. Defaults to False.
            residual_SAA_scenarios (int): Number of scenarios for residual SAA. Defaults to 1.
        """
        super().__init__(
            problem,
            learning_rate,
            device_str,
            predictor_str,
            decision_model_str,
            loss_function_str,
            to_decision_pars,
            use_dist_at_mode,
            False,
            standardize_predictions,
            init_OLS,
            seed,
            predictor_kwargs,
            None,
            decision_model_kwargs,
        )
        if loss_function_str == "smooth":
            assert decision_model_str == "quadratic", "Smooth loss works only with quadratic decision model!"
        self.batch_size = batch_size
        self.residual_SAA = residual_SAA
        self.residual_SAA_scenarios = residual_SAA_scenarios
        self._set_optimizer()
        self._set_loss_function()

    def _set_optimizer(self) -> None:
        """
        Initializes the Adam optimizer for training the trainable predictive model.
        """
        self.optimizer = torch.optim.Adam(self.trainable_predictive_model.parameters(), lr=self.learning_rate)

    def _set_loss_function(self, **kwargs) -> None:
        """
        Sets the loss function based on the specified loss_function_str.
        Supports various loss types including traditional losses and PyEPO differentiable losses.
        """
        if self.loss_function_str == "objective":
            self.loss_function = lambda *args, **kwargs: (self.problem.opt_model.model_sense_int * self.problem.opt_model.get_objective(*args, **kwargs))
            # self.loss_function = self.problem.opt_model.get_objective
        elif self.loss_function_str == "regret":
            self.loss_function = self.get_regret
        elif self.loss_function_str == "mse":
            self.loss_function = torch.nn.MSELoss()
        elif self.loss_function_str == "cave":
            self.add_binding_constraints_to_data()
            self.loss_function = InnerConeAlignedCosine(self.problem.opt_model)
        elif self.loss_function_str == "lava":
            threshold = kwargs.get("threshold", -0.1)
            self.add_adjacent_vertices_to_data(**kwargs)
            self.loss_function = LAVA(self.problem.opt_model, threshold=threshold)
        elif self.loss_function_str == "smooth":

            def smooth_loss(data_batch, decisions_batch, predictions_batch):
                decision = torch.cat([decisions_batch[var_name] for var_name in self.decision_model.var_names], dim=-1)
                prediction = torch.cat([predictions_batch[pred_var_name] for pred_var_name in self.decision_model.param_to_predict_names], dim=-1)
                constraint_violation = self.problem.opt_model.get_penalty(prediction, data_batch)
                mask_is_infeasible = (constraint_violation > 0)[:, None]
                reward_gradient = self.problem.opt_model.get_reward_gradient(decisions_batch, data_batch)
                radius_vector = decision - prediction
                dot_product = (reward_gradient * radius_vector).sum(-1, keepdim=True)
                radius_vector_norm_sq = (radius_vector * radius_vector).sum(-1, keepdim=True)
                projection = radius_vector * dot_product / radius_vector_norm_sq
                desired_direction = reward_gradient - mask_is_infeasible * projection
                loss = (-desired_direction.detach() * prediction).sum(-1)
                return loss

            self.loss_function = smooth_loss
        else:
            self.loss_function = PYEPO_LOSS_FUNCTIONS[self.loss_function_str](self.decision_model)
            if isinstance(self.loss_function, optModule):
                if self.problem.pyepo_solve_ratio < 1:
                    self.problem.ensure_solution_pool()
                    self.loss_function.solve_ratio = self.problem.pyepo_solve_ratio
                    self.loss_function.solpool = self.problem.solution_pool
                self.problem.register_pyepo_module(self.loss_function)

    def get_regret(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Computes the regret for given decisions compared to optimal decisions.
        Regret is defined as the difference between the objective value of the current
        decisions and the optimal decisions.

        Args:
            data_batch (dict[str, torch.Tensor]): Input data batch.
            decisions_batch (dict[str, torch.Tensor]): Current decisions to evaluate.
            predictions_batch (dict[str, torch.Tensor], optional): Predictions used for decisions. Defaults to None.

        Returns:
            torch.Tensor: The regret values for the batch.
        """
        objectives = self.problem.opt_model.get_objective(data_batch, decisions_batch, predictions_batch)
        optimal_decisions = self.problem.opt_model.solve_batch(data_batch)
        optimal_objectives = self.problem.opt_model.get_objective(data_batch, optimal_decisions)

        return self.problem.opt_model.model_sense_int * (objectives - optimal_objectives)

    def update(self, data_batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Performs a single training update step using the specified loss function.
        Computes predictions, decisions (if needed), loss, gradients, and updates parameters.

        Args:
            data_batch (dict[str, torch.tensor]): A batch of training data.

        Returns:
            dict[str, float]: Dictionary containing loss values, solver calls, gradient norm,
                             and evaluation metrics.
        """
        # Compute the predictions
        predictions_batch = self.predict(data_batch)
        # Compute decisions
        if self.loss_function_str in ["objective", "regret", "smooth"]:
            decisions_batch = self.decide(predictions_batch)
        else:
            decisions_batch = None
        # Compute losses
        loss = self.get_loss(data_batch, decisions_batch, predictions_batch)
        logger_loss = loss.detach().numpy().astype(np.float32)
        loss_mean = loss.mean()

        # Update
        self.optimizer.zero_grad()
        loss_mean.backward()
        # Save gradient norm
        grad_norm = 0
        for p in self.predictor.parameters():
            param_norm = p.grad.detach().data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5
        self.optimizer.step()

        batch_size = data_batch["features"].shape[0]
        if self.loss_function_str in ["objective", "regret", "smooth"]:
            batch_solver_calls = batch_size
        elif self.loss_function_str == "SPOPlus":
            batch_solver_calls = self.problem.pyepo_solve_ratio * batch_size
        else:
            batch_solver_calls = 0

        log_dict = {
            "loss": logger_loss,
            "used_loss": logger_loss,
            "solver_calls": batch_solver_calls,
            "grad_norm": grad_norm,
        }

        evaluate_metrics = ["mse", "mae"]
        if self.loss_function_str in ["objective", "regret", "smooth"]:
            evaluate_metrics.extend(["objective", "abs_regret", "rel_regret", "sym_rel_regret"])
        eval_dict = self.problem.evaluate(
            data_batch,
            decisions_batch,
            predictions_batch=predictions_batch,
            metrics=evaluate_metrics,
        )
        log_dict.update(eval_dict)

        return log_dict

    def _get_batch_results(self, data_batch: dict[str, torch.Tensor], metrics: list[str] | None = None) -> dict[str, Any]:
        """
        Processes a single batch of data to get predictions, decisions, and evaluation metrics.
        This extends the base implementation with support for used_loss.
        """
        predictions_batch = self.predict(data_batch)

        metrics = metrics or []
        needs_decisions = any(m in metrics for m in ["objective", "abs_regret", "rel_regret", "sym_rel_regret", "used_loss"])
        batch_size = data_batch["features"].shape[0]

        decisions_batch = None
        loss_is_pyepo = isinstance(self.loss_function, optModule)
        if needs_decisions:
            if loss_is_pyepo:
                decisions_batch = self._get_pyepo_decisions(predictions_batch)
            else:
                decisions_batch = self.decide(predictions_batch)

        batch_results = self.problem.evaluate(
            data_batch,
            decisions_batch,
            predictions_batch=predictions_batch,
            metrics=metrics,
        )
        if loss_is_pyepo and decisions_batch is not None:
            if self.problem.cache_at_val:
                batch_results["solver_calls"] = 0
            elif self.problem.pyepo_solve_ratio < 1:
                batch_results["solver_calls"] = self.problem.pyepo_solve_ratio * batch_size
            else:
                batch_results["solver_calls"] = batch_size
        else:
            batch_results["solver_calls"] = batch_size if decisions_batch is not None else 0

        for key in predictions_batch:
            batch_results[key] = predictions_batch[key].cpu().detach().numpy().astype(np.float32)
        if decisions_batch is not None:
            for key in decisions_batch:
                batch_results[key] = decisions_batch[key].cpu().detach().numpy().astype(np.float32)

        if "used_loss" in metrics:
            loss = self.get_loss(data_batch, decisions_batch, predictions_batch)
            batch_results["used_loss"] = loss.cpu().detach().numpy().astype(np.float32)

        return batch_results

    def get_loss(
        self,
        data_batch: dict[str, torch.Tensor],
        decisions_batch: dict[str, torch.Tensor],
        predictions_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Computes the loss value based on the specified loss function.
        Handles different loss types including objective, regret, MSE, smooth, and PyEPO losses.

        Args:
            data_batch (dict[str, torch.Tensor]): Input data batch.
            decisions_batch (dict[str, torch.Tensor]): Decision variables.
            predictions_batch (dict[str, torch.Tensor]): Predicted parameters.

        Returns:
            torch.Tensor: The computed loss value.
        """
        if (self.loss_function_str == "objective") or (self.loss_function_str == "regret"):
            return self.loss_function(data_batch, decisions_batch, predictions_batch)
        if self.loss_function_str == "smooth":
            return self.loss_function(data_batch, decisions_batch, predictions_batch)
        if self.loss_function_str == "cave":
            predictions = self.dict_to_predictions(predictions_batch)
            bctr = data_batch.get("bctr")
            if bctr is None:
                raise ValueError("CAVE loss requires 'bctr' in data_batch.")
            return self.loss_function(predictions, bctr)
        if self.loss_function_str == "lava":
            predictions = self.dict_to_predictions(predictions_batch)
            adjver = data_batch.get("adjver")
            if adjver is None:
                raise ValueError("LAVA loss requires 'adjver' in data_batch.")
            w_rel_list = []
            for key, shape in self.decision_model.var_shapes.items():
                relaxed_key = key + "_optimal_relaxed"
                if relaxed_key not in data_batch:
                    raise ValueError("LAVA loss requires optimal relaxed decisions in data_batch.")
                w_rel_list.append(data_batch[relaxed_key].reshape(-1, int(np.prod(shape))))
            w_rel = torch.cat(w_rel_list, dim=1)
            mm = -1 * self.problem.opt_model.model_sense_int
            return self.loss_function(predictions, adjver, w_rel, mm)
        if self.loss_function_str == "mse":
            total_loss = 0
            for key in predictions_batch:
                true_val = data_batch[key].to(torch.float).to(self.device)
                loss = self.loss_function(predictions_batch[key], true_val)
                total_loss += loss
            return total_loss

        predictions = self.dict_to_predictions(predictions_batch)
        if self.loss_function_str in [
            "blackboxOpt",
            "negativeIdentity",
            "perturbedOpt",
            "implicitMLE",
        ]:
            return self.loss_function(predictions)

        optimal_decisions = self.dict_to_decisions(data_batch, use_optimal=True)
        if self.loss_function_str == "perturbedFenchelYoung":
            return self.loss_function(predictions, optimal_decisions)

        if self.loss_function_str == "SPOPlus":
            true_parameters = self.dict_to_predictions(data_batch)
            optimal_decisions_batch = {}
            for key in self.decision_model.var_names:
                optimal_decisions_batch[key] = data_batch[key + "_optimal"]
            optimal_objectives = self.problem.get_objective(data_batch, optimal_decisions_batch, predictions_batch).view(-1, 1)
            self._solver_calls += self.problem.pyepo_solve_ratio * self.batch_size
            return self.loss_function(predictions, true_parameters, optimal_decisions, optimal_objectives)

        return NotImplementedError

    def add_binding_constraints_to_data(self) -> None:
        """
        Pre-computes and stores binding constraints for all train and validation instances.

        Solves the optimization problem for each instance in the combined train+validation set
        using `solve_batch_with_binding_constraints`, then writes the resulting constraint
        matrices into `self.problem.dataset` under the key `'bctr'`. Any previously stored
        `'bctr'` data is removed first. Raises `ValueError` if the optimization model does not
        support binding constraints (`supports_binding_constraints` must be True).
        """
        if "bctr" in self.problem.dataset.data_dict:
            self.problem.dataset.remove_data("bctr")
        if not getattr(self.problem.opt_model, "supports_binding_constraints", False):
            raise ValueError("CAVE loss requires an optimization model that supports binding constraints.")
        idx = self._get_train_and_val_idx()
        idx.sort()
        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            data = self.problem.read_data(batch_idx)
            _, bctr_list = self.problem.opt_model.solve_batch_with_binding_constraints(data)
            self.problem.dataset.add_data("bctr", bctr_list, indices=batch_idx)

    def add_adjacent_vertices_to_data(self) -> None:
        """
        Pre-computes and stores adjacent vertices (and the relaxed optimal solution) for all
        train and validation instances.

        For each instance in the combined train+validation set, enumerates the adjacent vertices
        of the optimal solution on the feasible polytope using `solve_batch_with_adjacent_vertices`
        and writes the results into `self.problem.dataset` under the key `'adjver'`. If
        `'optimal_relaxed'` is not yet present in the dataset, the LP relaxation of the model is
        also solved and stored under `'optimal_relaxed'`. If both keys already exist, this method
        returns immediately without recomputing. Raises `ValueError` if the optimization model does
        not support adjacent vertices (`supports_adjacent_vertices` must be True).
        """
        if "adjver" in self.problem.dataset.data_dict and "optimal_relaxed" in self.problem.dataset.data_dict:
            return
        if not getattr(self.problem.opt_model, "supports_adjacent_vertices", False):
            raise ValueError("LAVA loss requires an optimization model that supports adjacent vertices.")
        idx = self._get_train_and_val_idx()
        idx.sort()
        relaxed_model = None
        if "optimal_relaxed" not in self.problem.dataset.data_dict:
            relaxed_model = self.problem.opt_model.create_copy()
            if hasattr(relaxed_model, "relax_in_place"):
                relaxed_model.relax_in_place()

        for start in range(0, len(idx), self.batch_size):
            batch_idx = idx[start : start + self.batch_size]
            data = self.problem.read_data(batch_idx)
            _, adjver_list = self.problem.opt_model.solve_batch_with_adjacent_vertices(data)
            self.problem.dataset.add_data("adjver", adjver_list, indices=batch_idx)
            if relaxed_model is not None:
                optimal_relaxed = relaxed_model.solve_batch(data)
                self.problem.dataset.add_data("optimal_relaxed", optimal_relaxed, indices=batch_idx)

    def _flat_to_decisions_dict(self, flat_decisions: torch.Tensor) -> dict[str, torch.Tensor]:
        decisions_dict: dict[str, torch.Tensor] = {}
        idx = 0
        batch_size = flat_decisions.shape[0]
        for var_name, shape in self.decision_model.var_shapes.items():
            size = int(np.prod(shape))
            decisions_dict[var_name] = flat_decisions[:, idx : idx + size].reshape(batch_size, *shape)
            idx += size
        return decisions_dict

    def _get_pyepo_decisions(self, predictions_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        predictions = self.dict_to_predictions(predictions_batch).detach().cpu().numpy()
        if self.problem.cache_at_val:
            decisions_np, _ = _cache_in_pass(predictions, self.loss_function.optmodel, self.loss_function.solpool)
        elif self.problem.pyepo_solve_ratio < 1:
            decisions_np, _ = _solve_or_cache(predictions, self.loss_function)
        else:
            decisions_np, _ = _solve_in_pass(
                predictions,
                self.loss_function.optmodel,
                self.loss_function.processes,
                self.loss_function.solpool,
            )
            if self.problem.pyepo_solve_ratio < 1:
                self.loss_function._update_solution_pool(decisions_np)
        decisions_tensor = torch.as_tensor(
            decisions_np,
            device=next(iter(predictions_batch.values())).device,
            dtype=next(iter(predictions_batch.values())).dtype,
        )
        return self._flat_to_decisions_dict(decisions_tensor)

    def run_epoch(self, mode: str, epoch_num: int, metrics: list[str] | None = None) -> list[dict[str, float]]:
        """
        Runs one complete epoch in the specified mode (train/validation/test).
        Handles residual SAA updates if configured and processes all batches.

        Args:
            mode (str): The mode to run ('train', 'validation', or 'test').
            epoch_num (int): The current epoch number.
            metrics (list[str], optional): List of metrics to evaluate. Defaults to None.

        Returns:
            list[dict[str, float]]: List of dictionaries containing results for each batch.
        """
        assert mode in [
            "train",
            "validation",
            "test",
        ], "Mode must be train/validation/test!"
        if self.residual_SAA and (mode == self.use_dist_at_mode):
            error_samples = self.get_residuals()
            self.predictor.update_samples(error_samples)
            # We update the decision model, note that if we want this to work in validation we need to keep both models
            self.num_scenarios = self.residual_SAA_scenarios
            self.decision_model = self.problem.opt_model.create_saa_variant(self.num_scenarios)

        # Switch predictor and problem mode to train/evaluation
        self.trainable_predictive_model.train() if mode == "train" else self.trainable_predictive_model.eval()
        self.problem.set_mode(mode)

        # Initialize dictionary with the results
        epoch_results = []
        # Run
        for idx in self.problem.generate_batch_indices(self.batch_size):
            data_batch = self.problem.read_data(idx)
            if mode == "train":
                batch_results = self.update(data_batch)
            else:
                batch_results = self._get_batch_results(data_batch, metrics)
            mode_batch_results = {f"{mode}/{key}": val for key, val in batch_results.items()}
            mode_batch_results["batch_size"] = len(idx)
            epoch_results.append(mode_batch_results)

        return epoch_results

    def get_residuals(self):
        """
        Computes residuals (prediction errors) for residual SAA.
        Used when residual_SAA is enabled to gather error samples for improving predictions.

        Returns:
            list: List of residual tensors (prediction - true_values) for each sample.
        """
        errors = []
        # Note that the mode in problem is still set to the previous mode, which we want
        for idx in self.problem.generate_batch_indices(self.batch_size):
            data_batch = self.problem.read_data(idx)
            features = data_batch["features"].to(torch.float).to(self.device)
            predictions = self.predictor.forward(features)
            true_values = self.dict_to_predictions(data_batch)
            for i in range(len(idx)):
                errors.append(predictions[i] - true_values[i])

        return errors
