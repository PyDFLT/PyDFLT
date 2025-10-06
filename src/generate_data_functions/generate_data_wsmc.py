from typing import Optional

import numpy as np


def gen_data_wsmc(seed: Optional[int], num_data: int, num_features: int, num_items: int, degree: int = 1, noise_width: float = 0.0) -> dict[str, np.ndarray]:
    """
    Generate synthetic data for Weighted Set Multi-Cover (WSMC) problems.

    This function creates synthetic datasets for training WSMC optimization models.
    It generates feature vectors and corresponding coverage requirements using a
    polynomial relationship with optional noise.

    Args:
        seed (Optional[int]): Random seed for reproducible data generation.
        num_data (int): Number of data samples to generate.
        num_features (int): Dimensionality of feature vectors.
        num_items (int): Number of items in the WSMC problem.
        degree (int): Degree of polynomial relationship between features and coverage requirements. Defaults to 1.
        noise_width (float): Width of noise distribution applied to coverage requirements. Defaults to 0.0.

    Returns:
        dict[str, np.ndarray]: A dictionary containing:
            - 'coverage_requirements': Array of shape (num_data, num_items) with coverage requirements for each sample
            - 'features': Array of shape (num_data, num_features) with feature vectors for each sample

    Raises:
        ValueError: If degree is not a positive integer.
    """
    # Validate parameters
    if not isinstance(degree, int):
        raise ValueError(f"degree = {degree} should be int.")
    if degree <= 0:
        raise ValueError(f"degree = {degree} should be positive.")

    np.random.seed(seed)

    n = num_data
    p = num_features
    m = num_items

    # Random matrix parameter B
    B = np.random.binomial(1, 0.5, (m, p))
    # Feature vectors
    x = np.random.normal(0, 1, (n, p))
    # Value of items
    c = np.zeros((n, m), dtype=int)

    for i in range(n):
        # Cost without noise
        values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** degree + 1
        # Rescale
        values *= 5
        values /= 3.5**degree
        # Add noise
        epsilon = np.random.uniform(1 - noise_width, 1 + noise_width, m)
        values *= epsilon
        # Round to int
        values = np.ceil(values)
        c[i, :] = values

    c = c.astype(np.float32)

    data_dict = {"coverage_requirements": c, "features": x}
    return data_dict
