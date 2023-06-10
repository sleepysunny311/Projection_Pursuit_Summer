import argparse
import glob
import multiprocessing as mp
import os
import time
import yaml
import pickle as pkl
import itertools
import hashlib
import json
import pandas as pd
import numpy as np

from data_generation import GaussianDataGenerator
from Pursuit_Algorithms import *

def get_parser():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config-file', type=str, default='configs/defualt_config.yaml', metavar= "FILE" ,help='path to config file')
    parser.add_argument("--output", type=str, help="Output path")
    return parser


def get_cfg(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_cfg(default_dict, input_dict):
    merged_dict = default_dict.copy()  # Start with default values.
    merged_dict.update(input_dict)  # Override with user-provided values.
    return merged_dict
    
    
def get_output_path(output_path, config_filename):
    if output_path is None:
        # output file will be a pickle file in the outputs folder
        output_path = os.path.join("outputs", config_filename.split("/")[-1].split(".")[0] + ".pkl")
    else:
        # output file will be a pickle file in the specified folder
        output_path = os.path.join(output_path, config_filename.split("/")[-1].split(".")[0] + ".pkl")
    return output_path

def generate_params_combinations(config):
    # Delete TEST.trial_num from config
    del config['TEST']['trial_num']
    # Convert all values to lists.
    lists = {k: v if isinstance(v, list) else [v] for k, v in config.items()}
    # Generate combinations.
    keys, values = zip(*lists.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations
    
def testing(config, model_folder_path):
    all_performance = []
    
    # Generate combinations of parameters for parameters that are lists and trial number
    trial_num = config['TEST']['trial_num']
    param_combinations = generate_params_combinations(config)
    
    for params in param_combinations:
        tmp_performance = run_trials(params)
        all_performance.extend(tmp_performance)
    
    df_results = pd.DataFrame(all_performance)
    return df_results

def generate_hash(dictionary):
    # Convert dictionary to JSON string
    json_str = json.dumps(dictionary, sort_keys=True)

    # Generate hash from JSON string
    hash_object = hashlib.md5(json_str.encode())
    hash_value = hash_object.hexdigest()

    return hash_value


def run_one_trial(params, seed):
    """
    This function uses given parameter to generate needed data
    Match needed Algorithms and conduct research

    Cases in the trials:
    1. MP: Matching Pursuit
    2. OMP: Orthogonal Matching Pursuit
    3. BMP: Bagging Matching Pursuit
    4. BOMP: Bagging Orthogonal Matching Pursuit
    """
    TEST_params = params["TEST"]
    MODEL_params = params["MODEL"]


    Data_Geneartor = GaussianDataGenerator(TEST_params["N"],TEST_params["d"], TEST_params["true_sparsity"],TEST_params["noise_level"],seed)
    signal, true_indices, true_coefficients, perturbed_signal = Data_Geneartor.shuffle()

    Tmp_Model = None
    

    return 

def cal_performance(results):
    # TODO: Calculate and return a dictionary of the performance
    return

def run_trials(params, trial_num):
    """
    Run the trial for the given parameters for trial_num times
    """
    
    ###TODO: We should find those hyper parameters where are lists and put everything in the pool,
    ### This can be done with pandas


    params_trials_performance = []
    
    # Create Memory folder if not exists ./memory
    if not os.path.exists("./memory"):
        os.makedirs("./memory")
        
    # Genrate a hash for the current parameters to use it as the folder name to store the results
    params_hash = generate_hash(params)
    params_folder_path = os.path.join("./memory", params_hash)
        
    # Check if the trial has been done
    if os.path.exists(params_folder_path):
        print("This trial has been done before")
        # TODO: Load the results from the path
    else:
        # Create the folder using path
        os.makedirs(params_folder_path)
    
    # Run the trials and save the results in the folder 
    for i in range(trial_num):
        # Run the trial for the given parameters
        res_one_trail_tmp = run_one_trial(params, seed = i)
        # Save the results in the folder as a pickle file
        with open(os.path.join(params_folder_path, "trial_" + str(i) + ".pkl"), 'wb') as f:
            pkl.dump(res_one_trail_tmp, f)
        # Append the performance of the current trial to the list of all trials
        performance_tmp = cal_performance(res_one_trail_tmp)
        
        # ? Do we need to save the performance of each trial?
        params_trials_performance.append(performance_tmp)
    
    return params_trials_performance



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Merge default config and input config
    default_config = get_cfg("configs/default_config.yaml")
    input_config = get_cfg(args.config_file)
    config = merge_cfg(default_config, input_config)
    
    # Output folder for the current config file
    output_dir = args.output
    if output_dir is None:
        # output file will be a pickle file in the outputs folder
        output_dir = os.path.join("outputs", args.config_file.split("/")[-1].split(".")[0])
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # output file will be a pickle file in the specified folder
        output_dir = os.path.join(output_dir, args.config_file.split("/")[-1].split(".")[0])

    df_performance = testing(config, output_dir)
    
    with open(os.path.join(output_dir, "performance_results.pkl"), 'wb') as f:
        pkl.dump(df_performance, f)
        
    print("Done!")
    print("Results are saved in: ", output_dir)
    
    
    
    
