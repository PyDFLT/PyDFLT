import sys
import os
sys.path.append(os.getcwd())  # append the current working directory to the Python path

import yaml
from src.pydflt.utils.experiments import run, update_config
""" This script is for experiments for Sufficient Decision Proxies for DFL"""

experiment_kwargs = {
    'qp': {
        'decision_maker': {
            'decision_model_str': 'quadratic',
            'num_samples': 2,
        }
    },
    'qp_1': {
        'decision_maker': {
            'decision_model_str': 'quadratic',
            'num_samples': 1,
        }
    },
    'point': {
    },
    '2_point': {
        'decision_maker': {
            'decision_model_str': 'SAA',
            'decision_model_kwargs': {
                'num_scenarios': 2
            }
        }
    },
    '8_point': {
        'decision_maker': {
            'decision_model_str': 'SAA',
            'decision_model_kwargs': {
                'num_scenarios': 8
            }
        }
    },
    'pfl': {
        'decision_maker': {
            'name': 'PFL',
            'loss_function_str': 'mse',
        },
        'runner': {
            'main_metric': 'mse',
            'val_metrics': ['mse', 'objective', 'abs_regret', 'rel_regret', 'sym_rel_regret'],
            'test_metrics': ['mse', 'objective', 'abs_regret', 'rel_regret', 'sym_rel_regret']
        }
    },
    'residual_SAA': {
        'decision_maker': {
            'name': 'PFL',
            'loss_function_str': 'mse',
            'residual_SAA': True,
            'residual_SAA_scenarios': 16,
            'predictor_str': 'Sample',
            'decision_model_str': 'SAA',
            'to_decision_pars': 'sample',
            'use_dist_at_mode': 'test',
            'decision_model_kwargs': {
                'num_scenarios': 1,
            },
        },
        'runner': {
            'main_metric': 'mse',
            'val_metrics': ['mse'],
            'test_metrics': ['objective', 'abs_regret', 'rel_regret', 'sym_rel_regret']
        },
    },
}

keys_with_randomization = ['runner', 'problem', 'decision_maker', 'data', 'model']
experiments_to_run = ['pfl', 'residual_SAA', 'point', '2_point', 'qp']
seeds = range(5) # TODO adjust
for experiment_name in experiments_to_run:
    if experiment_name in experiment_kwargs:
        kwargs = experiment_kwargs[experiment_name]
        for seed in seeds:
            if experiment_name in experiments_to_run:
                yaml_dir = "experiments/sufficient-decision-proxies-ijcai2026/configs/wsmc.yml"
                config = yaml.safe_load(open(yaml_dir))
                config['runner']['experiment_name'] = f'{experiment_name}'
                for key in keys_with_randomization:
                    config[key]['seed'] = seed
                updated_config = update_config(config, kwargs)
                run(updated_config)
