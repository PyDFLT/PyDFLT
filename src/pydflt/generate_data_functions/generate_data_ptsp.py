import numpy as np


def gen_data_ptsp(
    seed: int | None = None,
    num_data: int = 500,
    num_features: int = 5,
    num_customers: int = 10,
    degree: int = 1,
    noise_width: float = 0.0,
    scale: float = 0.3,
) -> dict[str, np.ndarray]:
    """
    Generate synthetic data for Probabilistic Traveling Salesperson Problem (PTSP).

    This function creates synthetic datasets for training PTSP optimization models.
    It generates feature vectors and corresponding city visit probabilities using a
    polynomial relationship with optional noise.

    Args:
        seed (int | None): Random seed for reproducible data generation. Defaults to None.
        num_data (int): Number of data samples to generate. Defaults to 500.
        num_features (int): Dimensionality of feature vectors. Defaults to 5.
        num_customers (int): Number of customers (cities) in the PTSP instance. Defaults to 10.
        degree (int): Degree of polynomial relationship between features and visit probabilities. Defaults to 1.
        noise_width (float): Probability of randomly flipping a visit entry to a Bernoulli draw. Defaults to 0.0.
        scale (float): Scale parameter for data generation. Defaults to 0.3.

    Returns:
        dict[str, np.ndarray]: A dictionary containing:
            - 'visit': Array of shape (num_data, num_customers) with visit probabilities for each sample.
            - 'features': Array of shape (num_data, num_features) with feature vectors for each sample.

    Raises:
        ValueError: If degree is not a positive integer.
    """
    if not isinstance(degree, int):
        raise ValueError(f"degree = {degree} should be int.")
    if degree <= 0:
        raise ValueError(f"degree = {degree} should be positive.")

    rng = np.random.default_rng(seed)

    n = num_data
    p = num_features
    m = num_customers

    # Random matrix parameter B
    B = rng.binomial(1, 0.5, (m, p))  # noqa: N806
    # Feature vectors
    x = rng.normal(0, 1, (n, p))
    # Visit requirement
    c = np.zeros((n, m))

    for i in range(n):
        values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** degree + 1
        values /= 3.5**degree
        c[i, :] = np.clip(values, 0, 2) / 2

    c = np.round(c)
    random_draw = rng.binomial(1, 0.25, c.shape).astype(np.float32)
    noisy_entries = rng.binomial(1, noise_width, c.shape)
    c = c * (1 - noisy_entries) + random_draw * noisy_entries

    data_dict = {"visit": c.astype(np.float32), "features": x.astype(np.float32)}
    return data_dict
