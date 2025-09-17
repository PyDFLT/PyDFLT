import pyepo


def gen_data_shortestpath(
    seed: int = None,
    num_data: int = 500,
    num_features: int = 5,
    grid: tuple[int, int] = (10, 10),
    polynomial_degree: int = 6,
    noise_width: float = 0.5,
):
    features, values = pyepo.data.shortestpath.genData(
        num_data=num_data,
        num_features=num_features,
        grid=grid,
        deg=polynomial_degree,
        noise_width=noise_width,
        seed=seed,
    )

    print(f"Shape of values: {values.shape}, features: {features.shape}.")
    data_dict = {"arc_costs": values, "features": features}
    return data_dict
