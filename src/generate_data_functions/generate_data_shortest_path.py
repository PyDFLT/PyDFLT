from typing import Optional

import numpy as np
import pyepo


def gen_data_shortest_path(
    seed: Optional[int] = None,
    num_data: int = 500,
    num_features: int = 5,
    grid: tuple[int, int] = (10, 10),
    polynomial_degree: int = 6,
    noise_width: float = 0.5,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic data for shortest path problems.

    This function creates synthetic datasets for training shortest path optimization models.
    It generates feature vectors and corresponding arc costs for a grid network using PyEPO's
    shortest path data generation utilities.

    Args:
        seed (Optional[int]): Random seed for reproducible data generation. Defaults to None.
        num_data (int): Number of data samples to generate. Defaults to 500.
        num_features (int): Dimensionality of feature vectors. Defaults to 5.
        grid (tuple[int, int]): Grid dimensions (rows, columns) for the shortest path network. Defaults to (10, 10).
        polynomial_degree (int): Degree of polynomial relationship between features and costs. Defaults to 6.
        noise_width (float): Width of noise distribution applied to costs. Defaults to 0.5.

    Returns:
        dict[str, np.ndarray]: A dictionary containing:
            - 'arc_costs': Array of shape (num_data, num_arcs) with arc costs for each sample
            - 'features': Array of shape (num_data, num_features) with feature vectors for each sample
    """
    features, values = pyepo.data.shortestpath.genData(
        num_data=num_data,
        num_features=num_features,
        grid=grid,
        deg=polynomial_degree,
        noise_width=noise_width,
        seed=seed,
    )

    data_dict = {"arc_costs": values, "features": features}
    return data_dict
