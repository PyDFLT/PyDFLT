import sys
import os
sys.path.append(os.getcwd())  # append the current working directory to the Python path

import yaml
import torch
from src.pydflt.utils.load import load_data_from_dict
from src.pydflt.utils.experiments import run, update_config
""" This script is for experiments for Sufficient Decision Proxies for DFL published at IJCAI 2026, by 
    Noah Schutte, Krzysztof Postek, Grigorii Veviurko, Neil Yorke-Smith"""

experiment_kwargs = {
    'point': {
        'decision_maker': {
            'standardize_predictions': False
        },
    },
    '2_point': {
        'decision_maker': {
            'decision_model_str': 'scenario_based',
            'standardize_predictions': False,
            'decision_model_kwargs': {
                'num_scenarios': 2
            },
        }
    },
    '8_point': {
        'decision_maker': {
            'decision_model_str': 'scenario_based',
            'standardize_predictions': False,
            'decision_model_kwargs': {
                'num_scenarios': 8
            },
        }
    },
    '16_point': {
        'decision_maker': {
            'decision_model_str': 'scenario_based',
            'standardize_predictions': False,
            'decision_model_kwargs': {
                'num_scenarios': 16
            }
        }
    },
    'qp': {
        'decision_maker': {
            'decision_model_str': 'quadratic',
            'loss_function_str': 'regret'
        },
    },
    'pfl': {
        'decision_maker': {
            'name': 'pfl',
            'loss_function_str': 'mse',
        },
        'runner': {
            'main_metric': 'mse',
            'val_metrics': ['mse', 'objective', 'abs_regret', 'rel_regret', 'sym_rel_regret'],
            'test_metrics': ['mse', 'objective', 'abs_regret', 'rel_regret', 'sym_rel_regret']
        }
    },
    'residual_SAA': {
        'decision_maker_str': 'pfl',
        'decision_maker': {
            'loss_function_str': 'mse',
            'residual_SAA': True,
            'residual_SAA_scenarios': 16,
            'predictor_str': 'Sample',
            'decision_model_str': 'scenario_based',
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
    }
}

keys_with_randomization = ['runner', 'problem', 'decision_maker']
num_relevant_security = 7
seeds = range(5) # TODO adjust
experiments_to_run = ['pfl', 'residual_SAA', 'qp', 'point', '2_point', '8_point', '16_point']
for experiment_name in experiments_to_run:
    if experiment_name in experiment_kwargs:
        kwargs = experiment_kwargs[experiment_name]
        for seed in seeds:
            config = yaml.safe_load(open("experiments/configs/diff_portfolio_ln.yml"))
            data_path = f'data/portfolio_10_{seed}.pkl'
            config['data']['path'] = data_path
            config['runner']['experiment_name'] = f'{experiment_name}'
            data = load_data_from_dict(data_path)
            relevant_data = data['c'][:int(data['features'].shape[0] * config['problem']['train_ratio'])]
            bank_return = float(torch.kthvalue(relevant_data, num_relevant_security, dim=1)[0].median())
            if experiment_name == '2_point':
                config['decision_maker']['predictor_kwargs']['shift'] = bank_return
            config['model']['bank_return'] = bank_return
            for key in keys_with_randomization:
                config[key]['seed'] = seed
            updated_config = update_config(config, kwargs)
            run(updated_config)
