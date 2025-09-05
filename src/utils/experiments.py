import copy
from typing import Any, Optional

from optuna.trial import Trial

from src.problem import Problem
from src.registries.data import get_data
from src.registries.decision_makers import make_decision_maker
from src.registries.models import make_model
from src.runner import Runner
from src.utils.reproducability import set_seeds


def run(config: dict[str, Any], optuna_trial: Optional[Trial] = None) -> float:
    """
    Executes a complete experiment run based on the provided configuration.

    This function initializes the optimization model, generates data, creates a problem instance,
    sets up the decision maker, and finally runs the experiment using the Runner.
    It also sets global random seeds if specified in the configuration.

    Args:
        config (dict[str, Any]): A dictionary containing the full configuration for the experiment,
                                 including details for 'model', 'data', 'problem', 'decision_maker',
                                 and 'runner'.
        optuna_trial (Optional[Trial]): An Optuna trial object. If provided, the method will report validation
                                        metrics to Optuna and check for pruning.
    """
    if config.get("seed", None) is not None:
        set_seeds(config["seed"])
    model, config["model"] = make_model(**config["model"])
    data_dict, config["data"] = get_data(**config["data"])
    problem = Problem(data_dict=data_dict, opt_model=model, **config["problem"])
    decision_maker, config["decision_maker"] = make_decision_maker(problem=problem, **config["decision_maker"])
    runner = Runner(decision_maker=decision_maker, config=config, **config["runner"])
    best_validation_results = runner.run(optuna_trial=optuna_trial)

    return best_validation_results


def update_config(base_config: dict[str, Any], updates_config: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively updates a deep copy of the base dictionary with values from the update dictionary,
    but only for keys that exist in both. If an empty dictionary is provided in `updates_config` for a nested key,
    the corresponding dictionary in the new config will be replaced entirely.

    Args:
        base_config (dict[str, Any]): The original dictionary to be copied and updated.
        updates_config (dict[str, Any]): The dictionary with values and nested dictionaries
                                         to update the new config.

    Returns:
        dict[str, Any]: A new dictionary, which is a deep copy of `base_config`
                        with the `updates_config` applied.
    """
    # Create a deep copy of the base_config to avoid modifying the original
    new_config = copy.deepcopy(base_config)

    for key, value in updates_config.items():
        # Check if the key exists in the new_config and if both values are dictionaries
        if isinstance(value, dict) and (key in new_config) and isinstance(new_config[key], dict):
            if len(value) == 0:
                # If an empty dict is given in updates_config, replace the current dict
                new_config[key] = {}
            else:
                # Recursively update the nested dictionary in the new config
                new_config[key] = update_config(new_config[key], value)
        else:
            # If the value is not a dictionary, directly replace the value in the new config
            new_config[key] = value

    return new_config
