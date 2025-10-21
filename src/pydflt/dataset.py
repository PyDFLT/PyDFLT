import copy
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset


class DFLDataset(Dataset):
    """
    DFLDataset class extends PyTorch's Dataset for decision-focused learning (DFL) scenarios.

    It manages a dictionary of data (`data_dict`) where keys represent different quantities
    (e.g., 'features', 'costs', 'weights', 'optimal_decision') and values are torch.Tensors.
    The class ensures that all data tensors have the same number of samples (first dimension)
    and are at least 2D. It provides methods to access data by index and to add new data.

    Attributes:
        data_dict (dict[str, torch.Tensor]): The core dictionary holding all dataset components.
            Keys are strings identifying the data type (e.g., 'features', 'costs'),
            and values are torch.Tensors. All tensors are guaranteed to have the same
            number of samples in their first dimension and are at least 2D.
        keys (list[str]): A list of keys currently present in `data_dict`.
        num_samples (int): The number of samples in the dataset, determined from the
            first dimension of the tensors in `data_dict`.
    """

    def __init__(
        self,
        data_dict: dict[str, Union[np.ndarray, torch.Tensor]],
    ):
        """
        Initializes the DFLDataset.

        Args:
            data_dict (dict[str, np.ndarray | torch.Tensor]): A dictionary where keys are string identifiers
                for data components (e.g., 'features', 'costs') and values are NumPy arrays or tensors.
                All NumPy arrays must have the same length for their first dimension (number of samples).
                These arrays will be converted to torch.Tensors.
        """
        # Save input parameters
        self.data_dict = data_dict
        self.keys = list(data_dict.keys())

        # Verify that data_dict has the same number of samples for all keys
        self.num_samples = len(self.data_dict[self.keys[0]])
        for key in self.keys:
            assert len(data_dict[key]) == self.num_samples, f"{key} and {self.keys[0]} have different number of samples!"

        # Make sure that data is stored as torch.tensors; make everything at least two-dimensional: (batch, *shape)
        for key in self.keys:
            if isinstance(self.data_dict[key], np.ndarray):
                self.data_dict[key] = torch.from_numpy(self.data_dict[key]).to(torch.float32)
            self.data_dict[key] = torch.atleast_2d(self.data_dict[key])

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: Union[int, slice, list[int], torch.Tensor, np.ndarray]) -> dict[str, torch.Tensor]:
        """
        Retrieves a sample or a batch of samples from the dataset by index. Supports various indexing types
        compatible with PyTorch tensor indexing, including single integer, slice, list of integers,
        or a boolean/long tensor.

        Args:
            idx (int | slice | list[int] | torch.Tensor | np.ndarray): The index or indices
                to retrieve. If `idx` is an integer, a single sample is returned (each tensor
                will have its first dimension removed or be a view). If `idx` is a slice,
                list, or tensor, a batch of samples is returned.

        Returns:
            dict[str, torch.Tensor]: A dictionary where keys are data field names (e.g., 'features', 'costs') and
                values are the corresponding torch.Tensor data for the requested sample(s).
        """
        if isinstance(idx, list):
            idx = torch.tensor(np.array(idx))

        sample = {key: self.data_dict[key][idx] for key in self.keys}
        return sample

    def add_data(self, key: str, data: torch.Tensor | dict[str, torch.Tensor]) -> None:
        """
        Adds new data component(s) to the dataset.

        If `data` is a single torch.Tensor, it's added to `self.data_dict` with `key` as its key.
        If `data` is a dictionary of torch.Tensors, each tensor is added, and `key` is appended to each of their
        original keys. All added tensors are ensured to be at least 2D and have the correct number of samples.

        Args:
            key (str): The primary key name if `data_to_add` is a single tensor,
                              or the suffix to append to keys if `data_to_add` is a dictionary.
            data (torch.Tensor | dict[str, torch.Tensor]): The data to add.
                Can be a single torch.Tensor or a dictionary of torch.Tensors.
                The first dimension of any tensor must match `self.num_samples`.
        """
        if isinstance(data, dict):
            for data_key in data.keys():
                assert data[data_key].shape[0] == self.num_samples, "Trying to add data with wrong number of samples."
                self.data_dict[f"{data_key}_{key}"] = torch.atleast_2d(data[data_key])
        elif isinstance(data, torch.Tensor):
            assert data.shape[0] == self.num_samples, "Trying to add data with different number of samples."
            self.data_dict[f"{key}"] = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        self.keys = list(self.data_dict.keys())

    def __copy__(self) -> "DFLDataset":
        """
        Creates a deep copy of the DFLDataset instance. This ensures that the `data_dict` and its contained tensors
        are new objects, not just references to the original dataset's data.

        Returns:
            DFLDataset: A new DFLDataset instance that is a deep copy of the original.
        """
        return DFLDataset(copy.deepcopy(self.data_dict))
