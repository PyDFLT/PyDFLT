from typing import Optional

import numpy as np
import pyepo


def gen_data_two_stage_knapsack(
    seed: Optional[int],
    num_data: int,
    num_features: int,
    num_items: int,
    dimension: int,
    polynomial_degree: int,
    noise_width: float,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic data for two-stage knapsack problems.

    This function creates synthetic datasets for training two-stage knapsack optimization models.
    It generates feature vectors and corresponding item weights using PyEPO's knapsack data
    generation utilities. Unlike the standard knapsack problem, this focuses on weights rather
    than values for two-stage optimization scenarios.

    Args:
        seed (Optional[int]): Random seed for reproducible data generation.
        num_data (int): Number of data samples to generate.
        num_features (int): Dimensionality of feature vectors.
        num_items (int): Number of items in the knapsack problem.
        dimension (int): Dimension of the knapsack constraint (e.g., 1 for single constraint).
        polynomial_degree (int): Degree of polynomial relationship between features and weights.
        noise_width (float): Width of noise distribution applied to weights.

    Returns:
        dict[str, np.ndarray]: A dictionary containing:
            - 'item_weights': Array of shape (num_data, num_items) with item weights for each sample
            - 'features': Array of shape (num_data, num_features) with feature vectors for each sample
    """
    values, features, weights = pyepo.data.knapsack.genData(
        num_data,
        num_features,
        num_items,
        dim=dimension,
        deg=polynomial_degree,
        noise_width=noise_width,
        seed=seed,
    )

    data_dict = {"item_weights": weights, "features": features}
    return data_dict
