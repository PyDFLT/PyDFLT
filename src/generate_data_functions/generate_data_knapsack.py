from typing import Optional, Union

import numpy as np
import pyepo
import torch


def gen_data_knapsack(
    seed: Optional[int] = None,
    num_data: int = 500,
    num_features: int = 5,
    num_items: int = 10,
    dimension: int = 1,
    polynomial_degree: int = 6,
    noise_width: float = 0.5,
    torch_tensors: bool = False,
) -> dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Generate synthetic data for knapsack problems.

    This function creates synthetic datasets for training knapsack optimization models.
    It generates feature vectors and corresponding item values using PyEPO's knapsack
    data generation utilities.

    Args:
        seed (Optional[int]): Random seed for reproducible data generation. Defaults to None.
        num_data (int): Number of data samples to generate. Defaults to 500.
        num_features (int): Dimensionality of feature vectors. Defaults to 5.
        num_items (int): Number of items in the knapsack problem. Defaults to 10.
        dimension (int): Dimension of the knapsack constraint (e.g., 1 for single constraint). Defaults to 1.
        polynomial_degree (int): Degree of polynomial relationship between features and values. Defaults to 6.
        noise_width (float): Width of noise distribution applied to values. Defaults to 0.5.
        torch_tensors (bool): Whether to return PyTorch tensors instead of NumPy arrays. Defaults to False.

    Returns:
        dict[str, Union[np.ndarray, torch.Tensor]]: A dictionary containing:
            - 'item_value': Array of shape (num_data, num_items) with item values for each sample
            - 'features': Array of shape (num_data, num_features) with feature vectors for each sample
    """
    weights, features, values = pyepo.data.knapsack.genData(
        num_data,
        num_features,
        num_items,
        dim=dimension,
        deg=polynomial_degree,
        noise_width=noise_width,
        seed=seed,
    )

    if torch_tensors:
        values = torch.tensor(values, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)

    data_dict = {"item_value": values, "features": features}
    return data_dict
