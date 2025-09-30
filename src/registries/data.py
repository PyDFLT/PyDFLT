"""
This module provides a centralized registry (bottom of script) for various data generation functions.
It allows for easy registration of data generation methods and provides a flexible way to retrieve and use them with
default or overridden parameters.

The core functionality includes:
- `register_data`: To add new data generation functions to the registry.
- `get_data`: To retrieve and execute a registered data generation function.
"""

import copy
from typing import Any, Callable, Tuple

from src.generate_data_functions import (
    gen_data_knapsack,
    gen_data_shortestpath,
    gen_data_wsmc,
)
from src.utils.load import load_data_from_dict

data_registry: dict[str, Tuple[Callable, dict[str, Any]]] = {}


def register_data(name: str, data_function: Callable, **params: Any) -> None:
    """
    Register a data generation function with a specific name and optional parameters.

    Args:
        name (str): The name to register the data function under.
        data_function (Callable): The data generation function itself.
        **params (Any): Arbitrary keyword arguments that will be passed to the data function
                        when it is called.
    """
    data_registry[name] = (data_function, params)


def get_data(name: str, **override_params: Any) -> Tuple[Any, dict[str, Any]]:
    """
    Initialize a data generation function by its registered name with default or overridden parameters.

    Args:
        name (str): The name of the registered data function.
        **override_params (Any): Keyword arguments to override the default parameters
                                 of the registered data function.

    Returns:
        Tuple[Any, dict[str, Any]]: A tuple containing:
            - The generated data.
            - A dictionary of the final parameters used for data generation.
    """
    print(f"Generating data using {name}")
    if name not in data_registry:
        raise ValueError(f"Model '{name}' is not registered.")
    data_function, params_registry = data_registry[name]

    # Deep copies to avoid overwriting the registries
    params = copy.deepcopy(params_registry)

    # Allow parameter overrides
    final_params = {**params, **override_params}
    return data_function(**final_params), final_params


# # # # # # # # # # # # # # # # # # # # # # # # DATA FUNCTIONS # # # # # # # # # # # # # # # # # # # # # # # #
register_data(
    name="load_data_from_dict",
    data_function=load_data_from_dict,
    path=None,
)

register_data(  # as introduced in (Elmachtoub 2022)
    "shortest_path",
    gen_data_shortestpath,
    seed=5,
    num_data=2000,
    num_features=5,
    grid=(5, 5),
    polynomial_degree=6,
    noise_width=0.5,
)

# Parameters for "shortest_path" in different works:
# Elmachtoub 2022: num_data \in [100+25+10000, 1000+250+10000, 5000+1250+10000] (train+validation+test),
# num_features = 5, grid = (5, 5), polynomial_degree \in [1, 2, 4, 6, 8], noise_width \in [0, 0.5]

# Tang 2022: num_data \in [100+1000, 1000+1000, 5000+1000] (train+test), num_features = 5, grid = (5, 5),
# polynomial_degree \in [1, 2, 4, 6], noise_width \in [0, 0.5]

# Schutte 2024: num_data \in [100+100+1000, 1000+100+1000] (train+validation+test), num_features = 5, grid = (10, 10),
# polynomial_degree = 6, noise_width \in [0, 0.5, 1.0]

register_data(
    "knapsack",
    gen_data_knapsack,
    seed=5,
    num_data=2000,
    num_features=5,
    num_items=10,
    dimension=2,
    polynomial_degree=6,
    noise_width=0.5,
)

# Parameters for "knapsack" in different works:
# Tang 2022: num_data \in [100+1000, 1000+1000, 5000+1000] (train+test), num_features = 5, num_items = 10,
# polynomial_degree \in [1, 2, 4, 6], noise_width \in [0, 0.5]


register_data(
    "WSMC_Silvestri2024",
    gen_data_wsmc,
    seed=5,
    num_data=2500,  # num_data = ? (train, validation, test split: 80%, 10%, 10%)
    num_features=5,
    num_items=10,
    degree=5,
    noise_width=0.5,
)


# References
# Elmachtoub, A. N.; and Grigas, P. 2022. Smart “Predict, then Optimize”. Management Science, 68(1): 9–26.
# Bo Tang and Elias B. Khalil. PyEPO: A PyTorch-based end-to-end predict-then-optimize library for linear and integer
# programming
# Schutte, N.; Postek, K.; and Yorke-Smith, N. 2024. Robust Losses for Decision-Focused Learning. In Proceedings of
# the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI’24, 4868–4875.
