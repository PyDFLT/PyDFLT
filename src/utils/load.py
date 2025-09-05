import pickle
from pathlib import Path
from typing import Any, Optional, Tuple


def print_registry(registry: dict[str, Tuple[Any, dict[str, Any]]], filter_word=""):
    """
    Prints the contents of a given registry, optionally filtering by a keyword.
    This function iterates through the items in a registry. For each item, it prints
    its name, the name of the class/function, and its associated parameters.

    Args:
        registry (dict[str, Tuple[Any, dict[str, Any]]]): The registry dictionary to list.
                                                            Expected format: {name: (class/function, params_dict)}.
        filter_word (str): An optional keyword to filter the displayed items. Only items
                           whose names contain this word will be printed.
    """
    for key, val in registry.items():
        if filter_word not in key:
            continue
        print("Name: ", key)
        print("\nClass/function name: ", val[0])
        print("\nParameters: ", val[1])


def load_log(log_file_path: Path) -> dict[int, dict[str, float]]:
    """
    Loads epoch-wise logged data from a text file.

    Args:
        log_file_path (Path): The path to the log file.

    Returns:
        dict[int, dict[str, float]]: A dictionary where keys are epoch numbers and values are
                                     dictionaries of metric names and their float values for that epoch.
    """
    logs = {}
    try:
        with open(log_file_path, "r") as f:
            current_epoch = None
            for line in f:
                line = line.strip()
                if line.startswith("Epoch"):
                    # Parse the epoch number
                    current_epoch = int(line.split()[1].strip(":"))
                    if current_epoch not in logs:
                        logs[current_epoch] = {}
                elif current_epoch is not None and ":" in line:
                    # Parse the metric key and value
                    key, value = line.split(":")
                    logs[current_epoch][key.strip()] = float(value.strip())
    except FileNotFoundError:
        print(f"Log file '{log_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while loading the log: {e}")

    return logs


def load_data_from_dict(path: Optional[str] = None) -> dict[str, Any]:
    """
    Loads a pickled data dictionary from a specified file path.
    This function opens a file in binary read mode, loads the data using pickle,
    and then prints the keys and the shape of the corresponding values in the
    loaded dictionary.

    Args:
        path: The full path to the pickle file containing the
            data dictionary.

    Returns:
        The loaded data dictionary, typically mapping string keys
        to numpy arrays or tensors.
    """
    print("path", path)
    assert path is not None, "Specify the path to data_dict!"

    with open(path, "rb") as f:
        data_dict = pickle.load(f)
    print("Loaded data from %s" % path)
    for key, value in data_dict.items():
        print(key, value.shape)

    return data_dict
