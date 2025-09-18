from typing import Union

import numpy as np
import torch

from src.decision_makers.base import DecisionMaker
from src.predictors import IMPLEMENTED_PREDICTORS
from src.problem import Problem

# from src.logger import get_logger
# logger = get_logger(__name__)


class LancerDecisionMaker(DecisionMaker):
    """
    LANCER decision maker implementation based on the paper:
    "Landscape Surrogate: Learning Decision Losses for Mathematical Optimization Under Partial Information"
    (NeurIPS 2023) by Zharmagambetov A. et al.

    The algorithm is loosely based on the actor-critic algorithm and trains both a predictive model
    and a surrogate model that mimics the regret loss. The surrogate model learns to approximate
    the decision loss landscape, enabling gradient-based learning even when the optimization problem
    is non-differentiable.

    Attributes:
        relative_regret (bool): Whether to use relative regret for loss computation.
        learning_rate_surrogate (float): Learning rate for the surrogate model.
        learning_rate_predictor (float): Learning rate for the predictor model.
        weight_decay_surrogate (float): Weight decay for surrogate model regularization.
        weight_decay_predictor (float): Weight decay for predictor model regularization.
        regularizer (float): Regularization strength for the surrogate model.
        batch_size_surrogate_update (int): Batch size for surrogate model updates.
        batch_size_predictor_update (int): Batch size for predictor model updates.
        max_iters_surrogate_update (int): Maximum iterations for surrogate model updates per epoch.
        max_iters_predictor_update (int): Maximum iterations for predictor model updates per epoch.
        use_replay_buffer (bool): Whether to use a replay buffer for training.
        surrogate_model_str (str): Type of surrogate model to use.
        surrogate_model_kwargs (dict): Additional arguments for surrogate model initialization.
    """

    allowed_losses: list[str] = ["lancer"]

    allowed_decision_models: list[str] = [
        "base",
    ]

    allowed_predictors: list[str] = [
        "LinearSKL",
        "MLP",
    ]

    def __init__(
        self,
        problem: Problem,
        learning_rate_predictor: float = 0.005,
        learning_rate_surrogate: float = 0.001,
        device_str: str = "cpu",
        predictor_str: str = "MLP",
        decision_model_str: str = "base",
        to_decision_pars: str = "none",
        use_dist_at_mode: str = "none",
        standardize_predictions: bool = True,
        init_OLS: bool = False,
        seed: Union[int, None] = None,
        predictor_kwargs: dict = None,
        decision_model_kwargs: dict = None,
        regularizer: float = 0.1,
        surrogate_model_str: str = "MLP",
        surrogate_model_kwargs: dict = None,
        batch_size_surrogate_update: int = 1024,
        batch_size_predictor_update: int = 128,
        max_iters_surrogate_update: int = 5,
        max_iters_predictor_update: int = 5,
        use_replay_buffer: bool = True,
        weight_decay_predictor: float = 0.01,
        weight_decay_surrogate: float = 0.01,
        relative_regret: bool = True,
        num_epochs_pretraining_surrogate: int = 100,
        pretraining_surrogate: bool = False,
        **kwargs,
    ):
        """
        Initializes the LancerDecisionMaker.

        Args:
            problem (Problem): The problem instance containing data and optimization model.
            learning_rate_predictor (float): Learning rate for the predictor model. Defaults to 0.005.
            learning_rate_surrogate (float): Learning rate for the surrogate model. Defaults to 0.001.
            device_str (str): Device for computations ('cpu' or 'cuda'). Defaults to 'cpu'.
            predictor_str (str): Type of predictor to use. Defaults to 'MLP'.
            decision_model_str (str): Type of decision model. Defaults to 'base'.
            to_decision_pars (str): Strategy for converting predictions to decision parameters. Defaults to 'none'.
            use_dist_at_mode (str): When to use distributional predictions. Defaults to 'none'.
            standardize_predictions (bool): Whether to standardize predictions. Defaults to True.
            init_OLS (bool): Whether to initialize with OLS. Defaults to False.
            seed (Union[int, None]): Random seed for reproducibility. Defaults to None.
            predictor_kwargs (dict): Additional arguments for predictor initialization. Defaults to None.
            decision_model_kwargs (dict): Additional arguments for decision model initialization. Defaults to None.
            regularizer (float): Regularization strength for the surrogate model. Defaults to 0.1.
            surrogate_model_str (str): Type of surrogate model ('MLP'). Defaults to 'MLP'.
            surrogate_model_kwargs (dict): Additional arguments for surrogate model. Defaults to None.
            batch_size_surrogate_update (int): Batch size for surrogate model updates. Defaults to 1024.
            batch_size_predictor_update (int): Batch size for predictor model updates. Defaults to 128.
            max_iters_surrogate_update (int): Maximum iterations for surrogate updates per epoch. Defaults to 5.
            max_iters_predictor_update (int): Maximum iterations for predictor updates per epoch. Defaults to 5.
            use_replay_buffer (bool): Whether to use a replay buffer for training. Defaults to True.
            weight_decay_predictor (float): Weight decay for predictor regularization. Defaults to 0.01.
            weight_decay_surrogate (float): Weight decay for surrogate regularization. Defaults to 0.01.
            relative_regret (bool): Whether to use relative regret for loss computation. Defaults to True.
            num_epochs_pretraining_surrogate (int): Number of epochs for surrogate pretraining. Defaults to 100.
            pretraining_surrogate (bool): Whether to pretrain the surrogate model. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            problem,
            learning_rate_predictor,
            device_str,
            predictor_str,
            decision_model_str,
            "lancer",
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
        self.relative_regret = relative_regret
        self.learning_rate_surrogate = learning_rate_surrogate
        self.weight_decay_surrogate = weight_decay_surrogate
        print(f"Set learning rate surrogate to {self.learning_rate_surrogate}")
        self.learning_rate_predictor = learning_rate_predictor
        print(f"Set learning rate predictor to {self.learning_rate_predictor}")
        self.weight_decay_predictor = weight_decay_predictor
        self.regularizer = regularizer
        self.batch_size_surrogate_update = batch_size_surrogate_update
        self.batch_size_predictor_update = batch_size_predictor_update
        self.max_iters_surrogate_update = max_iters_surrogate_update
        self.max_iters_predictor_update = max_iters_predictor_update
        self.use_replay_buffer = use_replay_buffer
        self.surrogate_model_str = surrogate_model_str
        if surrogate_model_kwargs is None:
            surrogate_model_kwargs = {}
        self.surrogate_model_kwargs = surrogate_model_kwargs

        self._set_optimizer()
        self._set_loss_functions()
        self._set_surrogate_model(pretraining_surrogate, num_epochs_pretraining_surrogate)

    def _set_surrogate_model(self, pretraining_surrogate: bool, max_iters: int):
        self.surrogate_model = IMPLEMENTED_PREDICTORS[self.surrogate_model_str](
            num_inputs=self.problem.num_predictions,
            num_outputs=1,
            **self.surrogate_model_kwargs,
        )
        self.optimizer_surrogate_model = torch.optim.Adam(
            self.surrogate_model.parameters(),
            lr=self.learning_rate_surrogate,
            weight_decay=self.weight_decay_surrogate,
        )
        if pretraining_surrogate:
            self._pretrain_surrogate_model(max_iters)

    def _set_optimizer(self):
        self.optimizer_predictor = torch.optim.Adam(
            self.trainable_predictive_model.parameters(),
            lr=self.learning_rate_predictor,
            weight_decay=self.weight_decay_predictor,
        )

    def _set_loss_functions(self):
        self.loss_predictor = torch.nn.MSELoss()
        self.loss_surrogate_model = torch.nn.MSELoss()

    def _pretrain_surrogate_model(self, max_iter: int):
        print(f"Start pretraining surrogate for {max_iter} epochs")
        self.problem.set_mode("train")
        data = self.problem.read_data(self.problem.mode_to_indices["train"])

        # Obtain predictions for all samples in dataset (no grad computation)
        features = data["features"].to(torch.float32).to(self.device)

        # Get predictions and decisions for all training data points
        self.trainable_predictive_model.eval()  # We are not training the predictor right now
        with torch.no_grad():
            predictions = self.predictor.forward(features)
        predictions_dict = self.predictions_to_dict(predictions)
        true_values = self.dict_to_tensor(data)
        decisions = self.decide(predictions_dict)

        # Compute losses
        objectives = self.problem.opt_model.get_objective(data, decisions, predictions_batch=predictions)
        optimal_objectives = data["objective_optimal"]

        # We use absolute regret here, alternatively one could consider relative regret, or only the objective
        losses = (objectives - optimal_objectives) * self.problem.opt_model.model_sense_int
        if self.relative_regret:
            losses = losses / optimal_objectives.size(0)

        # Step over w: learning surrogate loss model
        surrogate_loss = self.update_surrogate_model(predictions, true_values, losses, max_iter)

        print(f"After pretraining surrogate loss is {surrogate_loss}")

    def update_surrogate_model(self, predictions, true_values, objectives, max_iter):
        """
        Fitting LANCER model
        Arguments: predictions, true_values, objectives (from experience replay buffer)
        """  # Use keys to predict somehow to obtain predictions and true values
        # Compute nr batches based on nr samples / batch size
        # Iterate through iterations
        assert predictions.shape == true_values.shape
        assert true_values.shape[0] == objectives.shape[0]
        batch_size = self.batch_size_surrogate_update
        N = true_values.shape[0]
        n_batches = int(N / batch_size)
        total_iter = 0
        while total_iter < max_iter:
            random_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idx = random_indices[bi * batch_size : (bi + 1) * batch_size]
                true_value = true_values[idx]
                f_hat_batch = objectives[idx].view(-1, 1).to(torch.float32).to(self.device)
                pred_batch = predictions[idx]
                input_surrogate_loss = torch.square(true_value - pred_batch).to(torch.float32).to(self.device)
                surrogate_loss = self.surrogate_model.forward(input_surrogate_loss)
                self.optimizer_surrogate_model.zero_grad()
                loss = self.loss_surrogate_model(surrogate_loss, f_hat_batch)
                loss.backward()
                self.optimizer_surrogate_model.step()
                total_iter += 1
                # logger.info(f'Fitting lancer, itr:  {total_iter} , loss:  {torch.mean(loss)}')
                if total_iter >= max_iter:
                    break
        # TODO: GV -- are you sure you want to return the loss on the last batch only?
        return loss

    def update_predictor(self, data):
        """
        Fitting prediction model on the training data
        """
        self.surrogate_model.eval()
        self.trainable_predictive_model.train()
        features = data["features"]
        true_values = self.dict_to_tensor(data)
        assert features.shape[0] == true_values.shape[0]
        batch_size = self.batch_size_predictor_update
        N = features.shape[0]
        n_batches = int(N / batch_size)

        total_iter = 0
        while total_iter < self.max_iters_predictor_update:
            rand_indices = np.random.permutation(N)
            for bi in range(n_batches + 1):
                idxs = rand_indices[bi * batch_size : (bi + 1) * batch_size]
                true_values_batch = true_values[idxs].to(torch.float32).to(self.device)
                features_batch = features[idxs].to(torch.float32).to(self.device)
                predictions_batch = self.predictor.forward(features_batch).to(torch.float32).to(self.device)
                self.optimizer_predictor.zero_grad()
                input_surrogate_model = torch.square(true_values_batch - predictions_batch).to(torch.float32).to(self.device)
                lancer_loss = self.surrogate_model.forward(input_surrogate_model)
                predictor_loss = self.loss_predictor(predictions_batch, true_values_batch)
                total_loss = torch.mean(lancer_loss + self.regularizer * predictor_loss)
                total_loss.backward()
                self.optimizer_predictor.step()
                total_iter += 1
                # logger.info(f'Fitting predictor, itr:  {total_iter} , lancer loss:  {torch.mean(total_loss)}')
                if total_iter >= self.max_iters_predictor_update:
                    break
        # TODO: GV -- are you sure you want to return the loss on the last batch only?
        return total_loss

    def run_epoch(self, mode: str, epoch_num: int, metrics: list[str] = None) -> list[dict[str, float]]:
        assert mode in [
            "train",
            "validation",
            "test",
        ], "Mode must be train/validation/test!"
        self.problem.set_mode(mode)
        data = self.problem.read_data(self.problem.mode_to_indices[mode])

        epoch_results = []
        if mode == "train":
            # Obtain predictions for all samples in dataset (no grad computation)
            features = data["features"].to(torch.float32).to(self.device)

            # Get predictions and decisions for all training data points
            self.trainable_predictive_model.eval()  # We are not training the predictor right now
            with torch.no_grad():
                predictions = self.predictor.forward(features)
            predictions_dict = self.predictions_to_dict(predictions)
            true_values = self.dict_to_tensor(data)
            decisions = self.decide(predictions_dict)

            # Compute losses
            objectives = self.problem.opt_model.get_objective(data, decisions, predictions_batch=predictions)
            # We can use regret, or relative regret, or only the objective
            optimal_objectives = data["objective_optimal"]
            losses = (objectives - optimal_objectives) * self.problem.opt_model.model_sense_int
            absolute_regret = losses.detach().numpy().astype(np.float32)

            # Update replay buffer
            if epoch_num == 1 or not self.use_replay_buffer:
                self.predictions_for_lancer = predictions.clone()  # Directly assign the Tensor
                self.true_values_for_lancer = true_values.clone()  # Directly assign the Tensor
                self.losses_for_lancer = losses.clone()  # Directly assign the Tensor
            else:
                self.predictions_for_lancer = torch.cat((self.predictions_for_lancer, predictions), dim=0)  # Concatenate along the 0th dimension
                self.true_values_for_lancer = torch.cat((self.true_values_for_lancer, true_values), dim=0)  # Concatenate along the 0th dimension
                self.losses_for_lancer = torch.cat((self.losses_for_lancer, losses), dim=0)

            # Step over w: learning surrogate loss model
            surrogate_loss = self.update_surrogate_model(
                self.predictions_for_lancer,
                self.true_values_for_lancer,
                self.losses_for_lancer,
                max_iter=self.max_iters_surrogate_update,
            )

            # Step over theta: learning predictor
            predictor_loss = self.update_predictor(data)

            # Store train results
            results = {
                "train/solver_calls": self._solver_calls,
                "train/surrogate_loss": torch.mean(surrogate_loss).detach().numpy(),
                "train/predictor_loss": torch.mean(predictor_loss).detach().numpy(),
                "train/eval": absolute_regret,
            }
            epoch_results.append(results)

        elif mode == "validation" or mode == "test":
            size = self.problem.validation_size if mode == "validation" else self.problem.test_size
            for idx in self.problem.generate_batch_indices(size):
                data_batch = self.problem.read_data(idx)
                batch_results = self._get_batch_results(data_batch, metrics)
                mode_batch_results = {"%s/%s" % (mode, key): val for key, val in batch_results.items()}
                epoch_results.append(mode_batch_results)

        return epoch_results
