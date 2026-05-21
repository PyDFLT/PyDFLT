import random

import torch


def set_seeds(seed: int, full_reproducibility_GPUs: bool = False) -> None:
    """
    Sets random seed for different packages.

    Args:
        seed (int): Random seed.
        full_reproducibility_GPUs (bool): Set to True to enable full GPU reproducibility. Might impact performance.
    """

    # np.random.seed is the legacy NumPy RNG API and is not used in this codebase;
    # numpy randomness is handled via np.random.default_rng (stored as self.rng).
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs if multiple are present
        if full_reproducibility_GPUs:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
