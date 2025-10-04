import numpy as np
import pyepo


def gen_data_two_stage_knapsack(
    seed: int,
    num_data: int,
    num_features: int,
    num_items: int,
    dimension: int,
    polynomial_degree: int,
    noise_width: float,
) -> dict[str, np.array]:
    values, features, weights = pyepo.data.knapsack.genData(
        num_data,
        num_features,
        num_items,
        dim=dimension,
        deg=polynomial_degree,
        noise_width=noise_width,
        seed=seed,
    )

    # print(f"Shape of weights: {weights.shape}, features: {features.shape}.")
    data_dict = {"item_weights": weights, "features": features}

    return data_dict
