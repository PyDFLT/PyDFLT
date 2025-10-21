"""
This module provides a centralized registry (bottom of script) for various optimization models.
It allows for easy registration of different optimization problem implementations and provides a flexible way to
initialize and use them with default or overridden parameters.

The core functionality includes:
- `register_model`: To add new optimization models to the registry.
- `make_model`: To initialize and retrieve an instance of a registered optimization model.
"""

import copy
from typing import Any, Type

from pydflt.concrete_models import (
    CVXPYDiffKnapsackModel,
    GRBPYKnapsackModel,
    ShortestPath,
    TravelingSalesperson,
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


def make_model(name: str, **override_params: Any) -> tuple[Any, dict[str, Any]]:
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
register_model(
    name="knapsack_2D_Tang2022",
    model_class=GRBPYKnapsackModel,
    num_decisions=32,
    capacity=20,
    weights_lb=3,
    weights_ub=8,
    dimension=2,
    seed=5,  # unknown seed
)

register_model(name="shortest_path", model_class=ShortestPath, grid=(5, 5))

"""
Parameters for "shortest_path" in different works:
Elmachtoub2022: grid = (5, 5)
Tang2022: grid = (5, 5)
Schutte2024: grid = (10, 10)
"""

register_model(name="tsp", model_class=TravelingSalesperson, num_nodes=20)

"""
Parameters for "shortest_path" in different works:
Tang2022: num_nodes = 20
Schutte2024: num_nodes = 20
"""

register_model(
    name="WSMC_Silvestri2024",
    model_class=WeightedSetMultiCover,
    num_items=5,  # \in [5, 10]
    num_covers=25,  # \in [25, 50]
    penalty=5,  # \in [1, 5, 10]
    cover_costs_lb=1,
    cover_costs_ub=100,
    Silvestri2024=True,
    seed=5,
)

register_model(name="knapsack_continuous", model_class=CVXPYDiffKnapsackModel, num_decisions=10, capacity=20, weights_lb=3, weights_ub=8, dimension=1, seed=5)


register_model(
    name="knapsack_2_stage", model_class=TwoStageKnapsack, num_decision=10, capacity=20, penalty_add=0.1, penalty_remove=10, values_lb=3, values_ub=8, seed=5
)

register_model(
    name="WSMC_Schutte2025",
    model_class=WeightedSetMultiCover,
    num_items=5,
    num_covers=25,
    penalty=5,
    cover_costs_lb=5,
    cover_costs_ub=50,
    recovery_ratio=0.8,
    seed=5,
)


"""
References
Elmachtoub2022
Adam N. Elmachtoub and Paul Grigas. Smart “predict, then optimize”’. Management Science, 68:9–26, 2022.
doi:10.1287/mnsc.2020.3922.

Schutte2024
Noah Schutte, Krzysztof Postek, and Neil Yorke-Smith. Robust losses for decision-focused learning. In Proceedings of
the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI’24, pages 4868–4875, 2024.
doi:10.24963/ijcai.2024/538.

Silvestri2024
Mattia Silvestri, Senne Berden, Jayanta Mandi, Ali ˙Irfan Mahmuto˘gulları, Maxime Mulamba, Allegra De Filippo,
Tias Guns, and Michele Lombardi. Score function gradient estimation to widen the applicability of decision-focused
learning. CoRR, abs/2307.05213, 2023.
doi:10.48550/arXiv.2307.05213.

Tang2024
Bo Tang and Elias B. Khalil. Pyepo: a pytorch-based end-to-end predict-then-optimize library for linear and integer
programming. Mathematical Program-ming Computation, 16(3):297–335, 2024.
doi:https://doi.org/10.1007/s12532-024-00255-x.

"""
