import os
import sys

sys.path.append(os.getcwd())  # append the current working directory to the Python path

import yaml

from src.utils.experiments import run, update_config

experiment_kwargs = {
    "SPOPlus": {
        "decision_maker": {
            "name": "pfl",
            "loss_function_str": "SPOPlus",
        },
    },
    "pfl": {},
}

seeds = list(range(1))
keys_with_randomization = ["runner", "problem", "decision_maker", "data"]
experiments_to_run = ["pfl", "SPOPlus"]  # list(experiment_kwargs.keys())
for experiment_name in experiments_to_run:
    if experiment_name in experiment_kwargs:
        for seed in seeds:
            kwargs = experiment_kwargs[experiment_name]
            yaml_dir = "experiments/configs/knapsack.yml"
            config = yaml.safe_load(open(yaml_dir))
            config["runner"]["experiment_name"] = f"{experiment_name}_100_n1"
            for key in keys_with_randomization:
                config[key]["seed"] = seed
            updated_config = update_config(config, kwargs)
            run(updated_config)
