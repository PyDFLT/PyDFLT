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

from utils.load import load_data_from_dict

from src.generate_data_functions import (
    gen_data_knapsack,
    gen_data_shortestpath,
    gen_data_wsmc,
)

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

register_data(
    "knapsack_pyepo",
    gen_data_knapsack,
    seed=5,
    num_data=1000,
    num_features=5,
    num_items=10,
    dimension=2,
    polynomial_degree=6,
    noise_width=0.5,
)

register_data(
    "shortestpath_5x5",
    gen_data_shortestpath,
    seed=5,
    num_data=500,
    num_features=5,
    grid=(5, 5),
    polynomial_degree=5,
    noise_width=0.5,
)

register_data(
    "wsmc_5x25",
    gen_data_wsmc,
    seed=5,
    num_data=2500,
    num_features=5,
    num_items=5,
    degree=5,
    noise_width=0.5,
)

register_data(
    "wsmc_10x50",
    gen_data_wsmc,
    seed=5,
    num_data=2500,
    num_features=5,
    num_items=10,
    degree=5,
    noise_width=0.5,
)
