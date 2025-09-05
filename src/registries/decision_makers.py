"""
This module provides a centralized registry (bottom of script) for various decision makers.
It allows for easy registration of different decision maker implementations and provides a flexible way to
initialize and use them with default or overridden parameters.

The core functionality includes:
- `register_decision_maker`: To add new decision makers to the registry.
- `make_decision_maker`: To initialize and retrieve an instance of a registered decision maker.
"""

import copy
from typing import Any, Tuple

from src.decision_makers import (
    DifferentiableDecisionMaker,
    SFGEDecisionMaker,
)

decision_maker_registry: dict[str, Tuple[type, dict[str, Any]]] = {}


def register_decision_maker(name: str, model_class: type, **params: Any) -> None:
    """
    Register a decision maker model with a specific name and optional parameters.

    Args:
        name (str): The name to register the model under.
        model_class (type): The class of the decision maker model.
        **params (Any): Arbitrary keyword arguments that will be passed to the model class
                        constructor when an instance is made.
    """
    decision_maker_registry[name] = (model_class, params)


def make_decision_maker(problem: Any, name: str, **override_params: Any) -> Tuple[Any, dict[str, Any]]:
    """
    Initialize a decision maker model by its registered name with default or overridden parameters.

    Args:
        problem (Any): The optimization problem instance that the decision maker will solve.
        name (str): The name of the registered decision maker model.
        **override_params (Any): Keyword arguments to override the default parameters
                                 of the registered decision maker model.

    Returns:
        Tuple[Any, dict[str, Any]]: A tuple containing:
            - An instance of the decision maker model.
            - A dictionary of the final parameters used for model initialization.
    """
    if name not in decision_maker_registry:
        raise ValueError(f"Model '{name}' is not registered.")

    # Deep copies to avoid overwriting the registries
    model_class, params_registry = decision_maker_registry[name]
    params = copy.deepcopy(params_registry)

    # Allow parameter overrides
    final_params = {**params, **override_params}

    return model_class(problem=problem, **final_params), final_params


# # # # # # # # # # # # # # # # # # # # # # # # DECISION MAKERS # # # # # # # # # # # # # # # # # # # # # # # #

register_decision_maker(
    name="sfge",
    model_class=SFGEDecisionMaker,
    learning_rate=0.01,
    batch_size=32,
    device_str="cpu",
    loss_function_str="sfge",
    predictor_str="MLP",
    predictor_kwargs={"bias": True},
    noisifier_kwargs={"bias": True, "sigma_setting": "independent"},
)

register_decision_maker(
    name="sfge_warmstart",
    model_class=SFGEDecisionMaker,
    learning_rate=0.01,
    batch_size=32,
    device_str="cpu",
    loss_function_str="sfge",
    predictor_str="MLP",
    predictor_kwargs={"bias": True, "init_bias_setting": "OLS"},
    noisifier_kwargs={"bias": True, "sigma_setting": "independent"},
)

register_decision_maker(
    name="differentiable",
    model_class=DifferentiableDecisionMaker,
    learning_rate=0.001,
    device_str="cpu",
    loss_function_str="regret",
    predictor_str="MLP",
)

register_decision_maker(
    name="pfl",
    model_class=DifferentiableDecisionMaker,
    learning_rate=0.001,
    device_str="cpu",
    loss_function_str="mse",
    predictor_str="MLP",
    predictor_kwargs={"n_layers": 2, "size": 256},
)
