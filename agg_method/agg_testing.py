import argparse
import glob
import multiprocessing as mp
import os
import time
import yaml
import hashlib
import json
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime

import numpy as np
from data_generation import GaussianDataGenerator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
import pickle as pkl

from algorithms import BMP
from algorithms2 import BOMP

import warnings
warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, default='configs/bomp_default.yaml', metavar= "FILE" ,help='path to config file')
    parser.add_argument("--output", type=str, help="Output path")
    return parser


def get_cfg(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_cfg(default_dict, input_dict):
    merged_dict = default_dict.copy()  # Start with default values.
    sections = ['MODEL', 'TEST']  # Specify sections to update

    for section in sections:
        if section in default_dict and section in input_dict:
            for key in default_dict[section]:
                # Check if the key is in the user input dictionary
                if key in input_dict[section]:
                    # If it is, update the merged dictionary
                    merged_dict[section][key] = input_dict[section][key]
                else:
                    # If not, print a message about using the default value
                    print(f"Missing parameter '{key}' in section '{section}', default value '{default_dict[section][key]}' will be used.")
        else:
            print(f"Missing section '{section}' in the user input, default values will be used.")

    # Check for invalid keys in the user input dictionary
    for section in input_dict:
        if section in sections:
            for key in input_dict[section]:
                if key not in default_dict[section]:
                    print(f"Invalid key '{key}' in section '{section}'. This key will be ignored.")
    return merged_dict
    
def get_output_path(output_path, config_filename):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_path is None:
        output_path = "./memory"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # output file will be a pickle file in the specified folder
    output_path = os.path.join(output_path, config_filename.split("/")[-1].split(".")[0] + timestamp + ".pkl")
    return output_path    

def get_model_params(config):
    import numpy as np
    all_params = config['MODEL']
    param_grid = {}
    fixed_params = {}
    K_start, K_end, K_step = all_params['K_start'], all_params['K_end'], all_params['K_step']
    if K_start >= K_end:
        raise ValueError("K_start must be smaller than K_end")
    if K_step <= 0:
        raise ValueError("K_step must be positive")
    # Check if K_start, K_end, K_step are integers
    if not isinstance(K_start, int) or not isinstance(K_end, int) or not isinstance(K_step, int):
        raise ValueError("K_start, K_end, K_step must be integers")
    K_list = np.arange(K_start, K_end, K_step, dtype=int)
    # Check if the param is a list or a single value if it is a list save to param_grid or else save to fixed_params
    for param, value in all_params.items():
        if param in ['K_start', 'K_end', 'K_step']:
            continue
        if isinstance(value, list):
            param_grid[param] = value
        else:
            fixed_params[param] = value
    param_grid['K'] = K_list
    return fixed_params, param_grid
    
def run_trials_npm_multi_noise_lvl(n, p, m, noise_level_lst, model_name, fixed_params, param_grid, cv_num, trial_num):
    # get the model

    if model_name == "BMP": 
        model = BMP(**fixed_params)
        
    if model_name == 'BOMP':
        model = BOMP(**fixed_params)

    res_log_npm = {
        'parameters': {'n': n, 'p': p, 'm': m, 'noise_level_lst': noise_level_lst, 
                       'model_name': model_name, 'cv_num': cv_num, 'trial_num': trial_num, 
                       'param_grid': param_grid, 'fixed_params': fixed_params},
        'log': []
    }
    
    K_lst = param_grid['K']
    N_bag = fixed_params['N_bag']

    for noise_level in noise_level_lst:
        noise_log = []
        print("Fitting models under noise level: ", noise_level)
        for trial_id in range(trial_num):
            Data_Geneartor = GaussianDataGenerator(p, n, m, noise_level, trial_id)
            true_signal, dictionary, true_indices, true_coefficients, perturbed_signal = Data_Geneartor.shuffle()
            X_train, X_test, y_train, y_test = train_test_split(dictionary, perturbed_signal, test_size=0.2, random_state=trial_id) 
            
            K_coef_lst_mat = np.zeros((len(K_lst), N_bag, p))
            K_final_coef_mat = np.zeros((len(K_lst), p))
            for i, K in enumerate(K_lst):
                model.set_params(**{'K': K})
                model.fit(X_train, y_train)
                model_coef_lst = model.coefficients_lst
                model_final_coef = model.coefficients
                model_coef_lst_mat = np.stack(model_coef_lst, axis=0)
                K_coef_lst_mat[i, :, :] = model_coef_lst_mat
                K_final_coef_mat[i, :] = model_final_coef
                
            reslog_one_trial = {'noise_level': noise_level, 'trial': trial_id, 'K_lst': K_lst, 
                                'K_coef_lst_mat': K_coef_lst_mat,  'K_final_coef_mat': K_final_coef_mat,
                                'dataset': (X_train, X_test, y_train, y_test)}
            noise_log.append(reslog_one_trial)
            print("Trial: ", trial_id, " Finished!")
        res_log_npm['log'].append(noise_log)
    return res_log_npm

def run_tests(config):
    n_tmp = config['TEST']['n']
    p_tmp = config['TEST']['p']
    m_tmp = config['TEST']['m']
    noise_level_lst = config['TEST']['noise_level']
    model_name = config['TEST']['model']
    cv_num = config['TEST']['cv_num']
    trial_num = config['TEST']['trial_num']
    
    # Get n, p, m, noise_level combinations
    if not isinstance(n_tmp, list):
        n_tmp = [n_tmp]
    if not isinstance(p_tmp, list):
        p_tmp = [p_tmp]
    if not isinstance(m_tmp, list):
        m_tmp = [m_tmp]
    
    npm_lst = list(product(n_tmp, p_tmp, m_tmp))
    
    if not isinstance(noise_level_lst, list):
        noise_level_lst = [noise_level_lst]
    
    # Get model parameters
    fixed_params, param_grid = get_model_params(config)
    print("Fixed params:", fixed_params)
    print("Param grid:", param_grid)
    
    # Start running the tests
    ALL_LOGS = []
    
    for n, p, m in npm_lst:
        print(f"Running trials for n = {n}, p = {p}, m= {m}")
        reslog_npm = run_trials_npm_multi_noise_lvl(n, p, m, noise_level_lst, model_name, fixed_params, param_grid, cv_num, trial_num)
        ALL_LOGS.append(reslog_npm)
        
    return ALL_LOGS

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Merge default config and input config
    default_config = get_cfg("configs/bmp_default.yaml")
    input_config = get_cfg(args.config)
    full_config = merge_cfg(default_config, input_config)
    
    # Output folder for the current config file
    output_dir = get_output_path(args.output, args.config)

    ALL_LOGS = run_tests(full_config)
    
    with open(output_dir, 'wb') as f:
        pkl.dump(ALL_LOGS, f)
        
    print("Done!")
    print("Results are saved in: ", output_dir)