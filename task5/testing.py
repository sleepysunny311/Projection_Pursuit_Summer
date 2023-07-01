from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import hydra
from omegaconf import DictConfig
from itertools import product
import json
import hashlib
import os

"""
Please remember to add imported modules to the server
"""


from algorithms import *
from data_generation import *
from crossvalidation import *
from visualization import *

def hash_encode(dictionary):
    # Convert dictionary to JSON string
    json_str = json.dumps(dictionary, sort_keys=True)

    # Generate hash from JSON string
    hash_object = hashlib.md5(json_str.encode())
    hash_value = hash_object.hexdigest()

    return hash_value

def expand_dict(data):
    if isinstance(data, list):
        for value in data:
            yield value
    elif isinstance(data, dict):
        keys = data.keys()
        values = (expand_dict(data[key]) for key in keys)
        for combination in product(*values):
            yield dict(zip(keys, combination))
    else:
        yield data



def generate_params_combinations(config):
    """
    This function takes in a config dictionary and generate all possible combinations of parameters

    Args:
        config (dict): A dictionary of parameters
    
    Returns:
        param_combinations (list): A list of dictionaries of parameters
    """

    # Convert all values to lists.
    return list(expand_dict(config))




class General_Testing:

    def __init__(self, configs):
        self.configs = configs

    def run_one_combination(self,configs):        
        res_log = {
        'parameters':{
            'N': configs['N'],
            'd': configs['d'],
            'm': configs['m'],
            'noise_level_lst': configs['noise_levels'],
            'cv_num': configs['cv_num'],
            'trial_num': configs['trial_num'],
            'K_lst': configs['K_lst'],

        },
        'noise_level_best_K': [],
        'noise_level_lowest_MSE': [],
        'log': []
        }        
        output_filename = configs['filename']
        noise_level_best_K = []
        noise_level_lowest_MSE = []
        for noise_level in configs['noise_levels']:
            print("Cross validating K under noise level: ", noise_level)
            trials_best_K_tmp = []
            MSE_loweset_K_temp = []
            for trial in range(configs['trial_num']):
                Data_Geneartor = GaussianDataGenerator(
                    dictionary_length=configs['N'],
                    dictionary_dimensions=configs['d'],
                    indice_number=configs['m'],
                    noise_level=noise_level,
                    random_seed=trial
                )
                true_signal, dictionary, true_indices, true_coefficients, perturbed_signal = Data_Geneartor.shuffle()
                K_cv_error = []
                for K in configs['K_list']:
                    Model = BaggingPursuit(
                        N = configs['N'],
                        K = K,
                        method=configs['method'],
                        signal_bag_flag=configs['signal_bag_flag'],
                        atom_bag_percent=configs['atom_bag_percent'],
                        select_atom_percent=configs['select_atom_percent'],
                        replace_flag=configs['replace_flag'],
                        agg_func=configs['agg_func'],
                        random_seed=trial,
                        ignore_warning=True
                    )
                    K_cv_error.append(cal_cv_error(Model, configs['cv_num'], perturbed_signal, dictionary))
                lowest_error = np.min(K_cv_error)
                lowest_error_K = configs['K_lst'][np.argmin(K_cv_error)]
                trials_best_K_tmp.append(lowest_error_K)
                MSE_loweset_K_temp.append(lowest_error)
                print("Trial: ", trial, " Best K: ", lowest_error_K, " Lowest Error: ", lowest_error)
                log_tmp = {
                    'noise_level': noise_level,
                    'trial': trial,
                    'Lowest_Error': lowest_error,
                    'lowest_error_K': lowest_error_K,
                    'cv_error_lst': K_cv_error
                }
                res_log['log'].append(log_tmp)
            noise_level_best_K.append(np.mean(trials_best_K_tmp))
            noise_level_lowest_MSE.append(np.mean(MSE_loweset_K_temp))
            print("Average best K for noise level: ", noise_level, " is: ", np.mean(trials_best_K_tmp), " with MSE: ", np.mean(MSE_loweset_K_temp))
        res_log['noise_level_best_K'] = noise_level_best_K
        res_log['noise_level_lowest_MSE'] = noise_level_lowest_MSE

        with open( output_filename, 'wb') as f:
            pkl.dump(res_log, f)
        print("Finished!")
        print("Log file saved to: ", + output_filename)
        return noise_level_best_K, noise_level_lowest_MSE, res_log
    
    def testing_all_comb(self,configs):
        all_performance = []
        
        # Generate combinations of parameters for parameters that are lists and trial number
        trial_num = configs['trial_num']
        param_combinations = generate_params_combinations(configs)
        
        for params in param_combinations:
            temp_noise_level_best_K, temp_noise_level_lowest_MSE, temp_res_log = self.run_one_combination(params)
            makeplots(temp_res_log)


@hydra.main(version_base="1.2", config_path="Configs", config_name="default.yaml")
def main(configs: DictConfig):
    test_class = General_Testing(configs)


if __name__ == "__main__":
    main()