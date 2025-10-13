import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import optuna
import yaml

from src.utils.experiments import run, update_config


class SearchSpaceConfig:
    """Handler for hyperparameter search space configuration."""

    def __init__(self, config_path: str):
        """Load search space configuration from YAML file."""
        with open(config_path, "r") as f:
            # This config is of depth two: {'decision_maker': {'lr':...}, 'predictor': ...}
            self.config = yaml.safe_load(f)
        self.components = list(self.config.keys())
        self.keys = {key: comp for comp in self.components for key in self.config[comp].keys()}

    def suggest_value(self, trial: optuna.Trial, param_name: str, param_config: Dict[str, Any]):
        """Suggest a value for a parameter based on its configuration."""
        if param_config["type"] == "float":
            return trial.suggest_float(
                param_name,
                param_config["low"],
                param_config["high"],
                log=param_config.get("log", False),
            )
        elif param_config["type"] == "int":
            return trial.suggest_int(
                param_name,
                param_config["low"],
                param_config["high"],
                step=param_config.get("step", 1),
            )
        elif param_config["type"] == "categorical":
            return trial.suggest_categorical(param_name, param_config["choices"])
        else:
            raise ValueError(f"Unknown parameter type: {param_config['type']}")

    def get_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the hyperparameters for a trial."""
        config = {}
        for component in self.components:
            config[component] = {}
            for param_name, param_config in self.config[component].items():
                config[component][param_name] = self.suggest_value(trial, param_name, param_config)
        return config


def create_study(study_name, prunner=None, storage_url=None):
    """Create or load an existing study."""
    if storage_url is None:
        storage_url = f"sqlite:////{study_name}.db"
    print("Storage url:", storage_url)
    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        pruner=prunner,
        direction="minimize",
        load_if_exists=True,
    )


def save_progress(study, search_space: SearchSpaceConfig, output_dir):
    """Save the best configuration to a YAML file."""
    best_config = search_space.get_trial_config(study.best_trial)
    # Add metadata
    save_config = {
        "best_value": study.best_value,
        "best_iteration": study.best_trial.number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": best_config,
    }
    os.makedirs(output_dir, exist_ok=True)
    # Save to YAML
    output_file = os.path.join(output_dir, "best_config.yaml")
    with open(output_file, "w") as f:
        yaml.dump(save_config, f, default_flow_style=False)
    print(f"Best configuration saved to {output_file}")


def run_trial(trial, search_space: SearchSpaceConfig, base_config, seeds: list):
    assert len(seeds) > 0, "Provide at least one seed!"

    trial_config = search_space.get_trial_config(trial)
    per_seed_results: list[float] = []

    for seed_idx, seed in enumerate(seeds, start=1):
        config = update_config(base_config=base_config, updates_config=trial_config)

        config["seed"] = seed
        for key in config:
            if isinstance(config[key], dict) and "seed" in config[key]:
                config[key]["seed"] = seed

        result = run(config, optuna_trial=None)
        per_seed_results.append(float(result))

        if trial is not None:
            intermediate = float(np.mean(per_seed_results))
            trial.report(intermediate, seed_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(per_seed_results))
