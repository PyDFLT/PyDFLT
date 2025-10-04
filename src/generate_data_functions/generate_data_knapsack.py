import pyepo
import torch


def gen_data_knapsack(
    seed: int = None,
    num_data: int = 500,
    num_features: int = 5,
    num_items: int = 10,
    dimension: int = 1,
    polynomial_degree: int = 6,
    noise_width: float = 0.5,
    torch_tensors: bool = False,
):
    weights, features, values = pyepo.data.knapsack.genData(
        num_data,
        num_features,
        num_items,
        dim=dimension,
        deg=polynomial_degree,
        noise_width=noise_width,
        seed=seed,
    )
    # print(f"Shape of values: {values.shape}, features: {features.shape}.")

    if torch_tensors:
        values = torch.tensor(values, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
    data_dict = {"item_value": values, "features": features}

    return data_dict
