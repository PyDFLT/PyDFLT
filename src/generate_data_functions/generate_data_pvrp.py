import numpy as np


def gen_data_pvrp(seed, num_data, num_features, num_customers, degree=1, noise_width=0.0, scale=0.3):
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
    m = num_customers
    # random matrix parameter B
    B = np.random.binomial(1, 0.5, (m, p))
    # feature vectors
    x = np.random.normal(0, 1, (n, p))
    # value of items
    c = np.zeros((n, m))
    for i in range(n):
        # cost without noise
        values = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** degree + 1
        # rescale
        values *= 1
        values /= 3.5**degree
        # noise
        # epsilon = np.random.uniform(1 - noise_width, 1 + noise_width, m)
        c[i, :] = np.clip(values, 0, 2) / 2  # * epsilon

    c = np.round(c)
    random = 1.0 * np.random.binomial(1, 0.25, c.shape)
    noisy_entries = np.random.binomial(1, noise_width, c.shape)
    c = c * (1 - noisy_entries) + random * noisy_entries

    # cs = c.reshape(-1)
    # print(sum(cs >= 5))
    # print(sum(cs <= 1))
    data_dict = {"visit": c, "features": x}

    return data_dict
