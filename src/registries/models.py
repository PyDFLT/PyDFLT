"""
This module provides a centralized registry (bottom of script) for various optimization models.
It allows for easy registration of different optimization problem implementations and provides a flexible way to
initialize and use them with default or overridden parameters.

The core functionality includes:
- `register_model`: To add new optimization models to the registry.
- `make_model`: To initialize and retrieve an instance of a registered optimization model.
"""

import copy
from typing import Any, Tuple, Type

from src.concrete_models import (
    CVXPYDiffKnapsackModel,
    GRBPYKnapsackModel,
    ShortestPath,
    TwoStageKnapsack,
    WeightedSetMultiCover,
)

model_registry = {}


def register_model(name: str, model_class: Type, **params: Any) -> None:
    """
    Register an optimization model with a specific name and optional parameters.

    Args:
        name (str): The name to register the model under.
        model_class (Type): The class of the optimization model.
        **params (Any): Arbitrary keyword arguments that will be passed to the model class
                        constructor when an instance is made.
    """
    model_registry[name] = (model_class, params)


def make_model(name: str, **override_params: Any) -> Tuple[Any, dict[str, Any]]:
    """
    Initialize an optimization model by its registered name with default or overridden parameters.

    Args:
        name (str): The name of the registered model.
        **override_params (Any): Keyword arguments to override the default parameters
                                 of the registered model.

    Returns:
        Tuple[Any, dict[str, Any]]: A tuple containing:
            - An instance of the optimization model.
            - A dictionary of the final parameters used for model initialization.
    """
    if name not in model_registry:
        raise ValueError(f"Model '{name}' is not registered.")
    model_class, params_registry = model_registry[name]

    # Deep copies to avoid overwriting the registries
    params = copy.deepcopy(params_registry)
    # model_class = copy.deepcopy(model_class_registry)

    # Allow parameter overrides
    final_params = {**params, **override_params}

    return model_class(**final_params), final_params


# # # # # # # # # # # # # # # # # # # # # # # # REGISTER MODELS # # # # # # # # # # # # # # # # # # # # # # # #
# Models directly from literature ------------------------------------------------------------------------------------ #
register_model(
    name="cvxpy_knapsack_pyepo",
    model_class=CVXPYDiffKnapsackModel,
    num_decisions=10,
    capacity=8,
    values_lb=3,
    values_ub=8,
    dimension=2,
    seed=None,
)

register_model(
    name="grbpy_knapsack_pyepo",
    model_class=GRBPYKnapsackModel,
    num_decisions=10,
    capacity=20,
    weights_lb=3,
    weights_ub=8,
    dimension=2,
)

# TODO: Naming consistency, gbpy/cvx
register_model(name="shortestpath_5x5", model_class=ShortestPath, grid=(5, 5), num_scenarios=1)

# TODO: Check parameters
register_model(
    name="knapsack_2_stage",
    model_class=TwoStageKnapsack,
    num_decision=10,
    capacity=22,
    penalty_add=0.1,
    penalty_remove=10,
    values_lb=3.0,
    values_ub=8.0,
    seed=5,
)

# Relevant models ---------------------------------------------------------------------------------------------------- #
register_model(
    name="knapsack_2_stage",
    model_class=TwoStageKnapsack,
    num_decision=10,
    capacity=22,
    penalty_add=0.1,
    penalty_remove=10,
    values_lb=3.0,
    values_ub=8.0,
    seed=5,
)

register_model(
    name="wsmc_5x25",
    model_class=WeightedSetMultiCover,
    num_items=5,
    num_covers=25,
    penalty=5,
    cover_costs_lb=5,
    cover_costs_ub=50,
    seed=5,
    num_scenarios=1,
)

register_model(
    name="wsmc_recovery_5x25",
    model_class=WeightedSetMultiCover,
    num_items=5,
    num_covers=25,
    penalty=5,
    cover_costs_lb=5,
    cover_costs_ub=50,
    recovery_ratio=0.8,
    seed=5,
    num_scenarios=1,
)

register_model(
    name="knapsack_2D",
    model_class=GRBPYKnapsackModel,
    dimension=2,
    num_decisions=10,
    capacity=22,
    weights_lb=3.0,
    weights_ub=8.0,
    seed=5,
)
