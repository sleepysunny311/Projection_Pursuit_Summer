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
from itertools import product
from datetime import datetime

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
    # Delete TEST.trial_num from config
    del config['TEST']['trial_num']
    # Convert all values to lists.
    return list(expand_dict(config))

def testing_all_comb(config):
    all_performance = []
    
    # Generate combinations of parameters for parameters that are lists and trial number
    trial_num = config['TEST']['trial_num']
    param_combinations = generate_params_combinations(config)
    
    for params in param_combinations:
        tmp_performance = run_trials(params, trial_num)
        all_performance.extend(tmp_performance)
    
    df_results = pd.DataFrame(all_performance)
    return df_results

def hash_encode(dictionary):
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
    
    model_method = MODEL_params["method"]

    ### TODO: When we are generating yamls, we should assign none to sub_num and depth even if we are doing MP/OMP
    ### ? Why we need to generate yaml files?
    
    bagging_sub_mum = MODEL_params["bagging_sub_num"]
    depth = MODEL_params["depth"]
    signal_bag_flag = MODEL_params["signal_bag_flag"]
    signal_bag_percent = MODEL_params["signal_bag_percent"]
    atom_bag_percent = MODEL_params["atom_bag_percent"]
    select_atom_percent = MODEL_params["select_atom_percent"]
    replace_flag = MODEL_params["replace_flag"]
    agg_func = MODEL_params["agg_func"]


    Data_Geneartor = GaussianDataGenerator(TEST_params["N"],TEST_params["d"], TEST_params["true_sparsity"],TEST_params["noise_level"],seed)
    true_signal, dictionary, true_indices, true_coefficients, perturbed_signal = Data_Geneartor.shuffle()
    Tmp_Model = None
    match model_method:
        case "MP":
            Tmp_Model = AtomBaggingMatchingPursuit(depth, atom_bag_percent, select_atom_percent, seed)
        case "OMP":
            Tmp_Model = AtomBaggingOrthogonalMatchingPursuit(depth, atom_bag_percent, select_atom_percent, seed)
        case "BMP":
            Tmp_Model = BaggingPursuit(bagging_sub_mum, depth, "MP", signal_bag_flag, 
                                            signal_bag_percent, atom_bag_percent, select_atom_percent, replace_flag, agg_func, seed)
        case "BOMP":
            Tmp_Model = BaggingPursuit(bagging_sub_mum, depth, "OMP", signal_bag_flag,
                                                signal_bag_percent, atom_bag_percent, select_atom_percent, replace_flag, agg_func, seed)
        case _:
            raise ValueError(f"Model method {model_method} is not supported")
    final_a, final_c = Tmp_Model.fit(perturbed_signal, dictionary)
    
    res_dict = {"true_signal": true_signal, 
                "perturbed_signal": perturbed_signal,
                "dictionary": dictionary, 
                "true_indices": true_indices,
                "true_coefficients": true_coefficients,
                "final_a": final_a,
                "final_c": final_c,
                "model": Tmp_Model}
    return res_dict

def cal_performance(res_dict):
    # Calculate MSE from res_dict
    return np.mean((res_dict["true_signal"] - res_dict["final_a"])**2)

def check_memory(params, trial_num):
    if not os.path.exists("./memory"):
        os.makedirs("./memory")
        
    # Genrate a hash for the current parameters to use it as the folder name to store the results
    params_hash = hash_encode(params)
    params_folder_path = os.path.join("./memory", params_hash)
        
    # Check if the trial has been done
    if os.path.exists(params_folder_path):
        print("This trial has been done before, loading the results...")
        # Check how many trials have been done
        if len(os.listdir(params_folder_path)) < trial_num:
            print("The number of trials is less than the trial_num, running the rest of the trials...")
            more_trial = trial_num - len(os.listdir(params_folder_path))
        elif len(os.listdir(params_folder_path)) == trial_num:
            print("All trials have been done, loading the results...")
            more_trial = 0
        else:
            print("This trail has been done",len(os.listdir(params_folder_path)), "times, loading the results...")
            more_trial = 0
    else:
        more_trial = trial_num
        # Create the folder to store the results
        os.mkdir(params_folder_path)
    
    return params_folder_path, more_trial
        

def load_trial_results(path):
    mse_lst = []
    # Loop through all the files in the folder
    files = os.listdir(path)
    for file in files:
        with open(os.path.join(path, file), 'rb') as f:
            res_dict = pkl.load(f)
            mse_lst.append(cal_performance(res_dict))
    return mse_lst

def run_trials(params, trial_num):
    """
    Run the trial for the given parameters for trial_num times
    """
    
    param_folder_path, more_trial = check_memory(params, trial_num)
    
    if more_trial == 0:
        # Load all the results from the folder_path
        mean_mse = np.mean(load_trial_results(param_folder_path))
    else:
        trial_done = len(os.listdir(param_folder_path))
        # Run the rest of the trials
        MSE_lst = load_trial_results(param_folder_path)
        # Run the trials and save the results in the folder 
        for i in range(trial_done, trial_num):
            # Run the trial for the given parameters
            res_one_trail_dict = run_one_trial(params, seed = i)
            # Save the results in the folder as a pickle file
            with open(os.path.join(param_folder_path, "trial_" + str(i) + ".pkl"), 'wb') as f:
                pkl.dump(res_one_trail_dict, f)
            # Append the performance of the current trial to the list of all trials
            mse_tmp = cal_performance(res_one_trail_dict)
            MSE_lst.append(mse_tmp)
        mean_mse = np.mean(MSE_lst)
            
    # Combine parameters and mse performance
    performance_dict = params.copy()
    performance_dict["mse"] = mean_mse
    
    return performance_dict



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
        
    # Create the output folder if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # combine "Performance" and the time stamp as the file name
    performance_res_filename = "Performance_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pkl"

    df_performance = testing_all_comb(config)
    
    with open(os.path.join(output_dir, performance_res_filename), 'wb') as f:
        pkl.dump(df_performance, f)
        
    print("Done!")
    print("Results are saved in: ", output_dir + "/" + performance_res_filename)
    
    
    
    
