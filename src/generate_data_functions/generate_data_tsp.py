from typing import Optional, Union

import numpy as np
import pyepo
import pyepo.data.tsp
import torch


def gen_data_traveling_salesperson(
    seed: Optional[int] = None,
    num_data: int = 500,
    num_features: int = 5,
    num_nodes: int = 10,
    polynomial_degree: int = 6,
    noise_width: float = 0.5,
    torch_tensors: bool = False,
) -> dict[str, Union[np.ndarray, torch.Tensor]]:
    """
    Generate synthetic data for Traveling Salesperson Problem (TSP).

    This function creates synthetic datasets for training TSP optimization models.
    It generates feature vectors and corresponding edge costs for a complete graph
    using PyEPO's TSP data generation utilities.

    Args:
        seed (Optional[int]): Random seed for reproducible data generation. Defaults to None.
        num_data (int): Number of data samples to generate. Defaults to 500.
        num_features (int): Dimensionality of feature vectors. Defaults to 5.
        num_nodes (int): Number of nodes in the TSP instance. Defaults to 10.
        polynomial_degree (int): Degree of polynomial relationship between features and costs. Defaults to 6.
        noise_width (float): Width of noise distribution applied to costs. Defaults to 0.5.
        torch_tensors (bool): Whether to return PyTorch tensors instead of NumPy arrays. Defaults to False.

    Returns:
        dict[str, Union[np.ndarray, torch.Tensor]]: A dictionary containing:
            - 'edge_costs': Array of shape (num_data, num_edges) with edge costs for each sample, where num_edges = num_nodes * (num_nodes - 1) / 2
            - 'features': Array of shape (num_data, num_features) with feature vectors for each sample
    """
    features, values = pyepo.data.tsp.genData(num_data, num_features, num_nodes=num_nodes, deg=polynomial_degree, noise_width=noise_width, seed=seed)

    if torch_tensors:
        values = torch.tensor(values, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)

    data_dict = {"edge_costs": values, "features": features}
    return data_dict
