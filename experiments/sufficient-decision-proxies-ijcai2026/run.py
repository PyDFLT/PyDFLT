import sys
import os
sys.path.append(os.getcwd())

import argparse
from pathlib import Path

import numpy as np
import yaml

from pydflt.utils.load import load_data_from_npz
from src.pydflt.utils.experiments import run, update_config

"""
Run script for Sufficient Decision Proxies for DFL, IJCAI 2026, by
Noah Schutte, Krzysztof Postek, Grigorii Veviurko, Neil Yorke-Smith.

Usage (use --seeds argument for non-default seeds):
    python experiments/sufficient-decision-proxies-ijcai2026/run.py \\
        --problem portfolio \\
        --methods pfl residual_SAA qp point 2_point 8_point 16_point

    python experiments/sufficient-decision-proxies-ijcai2026/run.py \\
        --problem wsmc \\
        --methods pfl residual_SAA qp point 2_point

    python experiments/sufficient-decision-proxies-ijcai2026/run.py \\
        --problem ptsp \\
        --methods pfl residual_SAA qp point 2_point 8_point
"""

EXPERIMENT_DIR = Path("experiments/sufficient-decision-proxies-ijcai2026")
NUM_RELEVANT_SECURITY = 7
PORTFOLIO_DATA_PATH_TEMPLATE = "experiments/sufficient-decision-proxies-ijcai2026/data/portfolio_10_{seed}.npz"


def portfolio_pre_run_hook(config: dict, seed: int, method: str) -> dict:
    """
    For the portfolio problem, we set a bank return rate based on the median return of the relevant securities
    """
    data_path = PORTFOLIO_DATA_PATH_TEMPLATE.format(seed=seed)
    config["data"]["path"] = data_path
    data = load_data_from_npz(data_path)
    relevant_data = data["return"][: int(data["features"].shape[0] * config["problem"]["train_ratio"])]
    bank_return = float(np.median(np.partition(relevant_data, NUM_RELEVANT_SECURITY - 1, axis=1)[:, NUM_RELEVANT_SECURITY - 1]))
    config["model"]["bank_return"] = bank_return
    if method == "2_point":
        config["decision_maker"]["predictor_kwargs"]["shift"] = bank_return
    return config


def main():
    parser = argparse.ArgumentParser(description="Run IJCAI 2026 sufficient decision proxies experiments.")
    parser.add_argument("--problem", required=True, help="Problem name, e.g. wsmc, ptsp, portfolio")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods to run, e.g. point 2_point qp pfl residual_SAA")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5, 15)), help="Seeds to run, e.g. 5 6 7 8 9. Defaults to 5 through 14.")
    args = parser.parse_args()

    problem_config_path = EXPERIMENT_DIR / "configs" / "problems" / f"{args.problem}.yml"
    if not problem_config_path.exists():
        raise FileNotFoundError(f"Problem config not found: {problem_config_path}")

    for method in args.methods:
        method_config_path = EXPERIMENT_DIR / "configs" / "methods" / f"{method}.yml"
        if not method_config_path.exists():
            raise FileNotFoundError(f"Method config not found: {method_config_path}")
        method_config = yaml.safe_load(open(method_config_path)) or {}

        for seed in args.seeds:
            config = yaml.safe_load(open(problem_config_path))

            # Pop meta-keys before passing config to run()
            keys_with_randomization = config.pop("_keys_with_randomization", ["runner", "problem", "decision_maker", "data", "model"])
            method_overrides = config.pop("method_overrides", {})

            # Apply method config, then any problem-specific override for this method
            config = update_config(config, method_config)
            if method in method_overrides:
                config = update_config(config, method_overrides[method])

            # Set experiment name and seeds
            config["runner"]["experiment_name"] = method
            for key in keys_with_randomization:
                if key in config:
                    config[key]["seed"] = seed

            # Apply portfolio-specific hook for bank return computation
            if args.problem == "portfolio":
                config = portfolio_pre_run_hook(config, seed, method)

            run(config)


if __name__ == "__main__":
    main()
