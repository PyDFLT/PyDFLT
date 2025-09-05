import numpy as np


def gen_data_wsmc(seed, num_data, num_features, num_items, degree=1, noise_width=0.0):
    # positive integer parameter
    if not isinstance(degree, int):
        raise ValueError("degree = {} should be int.".format(degree))
    if degree <= 0:
        raise ValueError("degree = {} should be positive.".format(degree))
    # set seed
    np.random.seed(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # number of items
    m = num_items
    # random matrix parameter B
    B = np.random.binomial(1, 0.5, (m, p))
    # feature vectors
    x = np.random.normal(0, 1, (n, p))
    # value of items
    c = np.zeros((n, m), dtype=int)
    for i in range(n):
        # cost without noise
        values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** degree + 1
        # rescale
        values *= 5
        values /= 3.5**degree
        # noise
        epsilon = np.random.uniform(1 - noise_width, 1 + noise_width, m)
        values *= epsilon
        # convert into int
        values = np.ceil(values)
        c[i, :] = values
        # float
        c = c.astype(np.float32)

    data_dict = {"coverage_requirements": c, "features": x}

    return data_dict
