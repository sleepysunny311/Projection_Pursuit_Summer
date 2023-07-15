"""
This file serves as a baseline for the BOMP algorithm. It can read the config file and run 
the OMP algorithm with the parameters specified in the config file which is designed for BOMP.
"""

import argparse
import yaml
import numpy as np
import pickle as pkl
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from itertools import product
from datetime import datetime
import json
import hashlib
import os
from algorithms import OMP_Augmented
from data_generation import *


import warnings

warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config-name', type=str)
    parser.add_argument('--config-path', type=str)
    parser.add_argument('--output-path', type=str)
    return parser



def hash_encode(dictionary):
    # Convert dictionary to JSON string
    dictionary = dict(sorted(dictionary.items()))
    json_str = json.dumps(dictionary, sort_keys=True)

    # Generate hash from JSON string
    hash_object = hashlib.md5(json_str.encode())
    hash_value = hash_object.hexdigest()
    return hash_value

def clear_log(res_log_npm):
    res_log_npm["log"] = []
    res_log_npm["noise_level_lowest_cv_MSE"] = []
    res_log_npm["trials_testing_score"] = []
    return res_log_npm

def dump_single_res(res_log_npm, filename):
    local_file_exists = os.path.isfile(filename)
    local_log = None
    ready_to_dump = (len(res_log_npm['log'])>0) | (len(res_log_npm['noise_level_lowest_cv_MSE'])>0)
    if ready_to_dump:
        if local_file_exists:
            with open(filename, "rb") as f:
                local_log_lists = pkl.load(f)
                local_log = local_log_lists[-1]
                if local_log["hash"] == res_log_npm["hash"]:
                    local_log['log'] = local_log['log'] + res_log_npm['log']
                    local_log['noise_level_lowest_cv_MSE'] = local_log['noise_level_lowest_cv_MSE'] + res_log_npm['noise_level_lowest_cv_MSE']
                    local_log['trials_testing_score'] = local_log['trials_testing_score'] + res_log_npm['trials_testing_score']
                    local_log_lists[-1] = local_log
                else:
                    local_log_lists.append(res_log_npm)
        else:
            local_log_lists = [res_log_npm]
        with open(filename, "wb") as f:
            pkl.dump(local_log_lists, f)
        dumped_log = len(res_log_npm['log'])
        dumped_error = len(res_log_npm['noise_level_lowest_cv_MSE'])
        #print(f"Dumped {dumped_log} logs and {dumped_error} errors")
        res_log_npm = clear_log(res_log_npm)
    else:
        print("Nothing to dump")

    return res_log_npm

def get_model_params(config):
    OMP_arg_lst = ["K_lst", "select_atom_percent", "random_seed","ignore_warning"]
    all_params = config["MODEL"]
    param_grid = {}
    fixed_params = {}

    Bag_lst = all_params["Bag_lst"]
    K_lst = all_params["K_lst"]

    del all_params["Bag_lst"]
    del all_params["K_lst"]

    for param, value in all_params.items():
        if param in OMP_arg_lst:
            if isinstance(value, list):
                param_grid[param] = value
            else:
                fixed_params[param] = value

    fixed_params["K_lst"] = K_lst
    return fixed_params, param_grid


def run_trials_npm_multi_noise_lvl(
    n, p, m, noise_level_lst, model_name, fixed_params, param_grid, cv_num, trial_num, filename
):
    # get the model

    if model_name == "BOMP":
        model = OMP_Augmented(**fixed_params)

    res_log_npm = {
        "parameters": {
            "n": n,
            "p": p,
            "m": m,
            "noise_level_lst": noise_level_lst,
            "model_name": model_name,
            "cv_num": cv_num,
            "trial_num": trial_num,
            "param_grid": param_grid,
            "fixed_params": fixed_params,
        },
        "noise_level_lowest_cv_MSE": [],
        "trials_testing_score": [],
        "log": [],
    }
    # print(type(res_log_npm))
    # for key, value in res_log_npm.items():
    #     print(type(value))
    res_log_npm["hash"] = hash_encode(res_log_npm["parameters"])
    print(f"Running trials for n = {n}, p = {p}, m = {m}")
    for noise_level in noise_level_lst:
        print("Cross validating alpha under noise level: ", noise_level)
        trials_loweset_cv_MSE_temp = []
        trials_testing_score_temp = []
        for trial_id in range(trial_num):
            Data_Geneartor = GaussianDataGenerator(p, n, m, noise_level, trial_id)
            (
                true_signal,
                dictionary,
                true_indices,
                true_coefficients,
                perturbed_signal,
            ) = Data_Geneartor.shuffle()
            X_train, X_test, y_train, y_test = train_test_split(
                dictionary, perturbed_signal, test_size=0.2, random_state=trial_id
            )
            gs = GridSearchCV(
                model,
                param_grid,
                cv=cv_num,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train, y_train)
            cv_err_lst = -gs.cv_results_["mean_test_score"]
            param_lst = gs.cv_results_["params"]
            best_estimator = gs.best_estimator_
            old_bag_k_error_series = best_estimator.error_series
            best_estimator.set_K_lst([best_estimator.optimal_k])
            best_estimator.fit(X_train, y_train)
            testing_error = mean_squared_error(y_test, best_estimator.predict(X_test))
            trials_testing_score_temp.append(testing_error)
            lowest_cv_error = np.min(cv_err_lst)
            trials_loweset_cv_MSE_temp.append(lowest_cv_error)
            best_params = gs.best_params_
            best_params["best_k"] = best_estimator.optimal_k
            reslog_one_trial = {
                "noise_level": noise_level,
                "trial": trial_id,
                "cv_error_lst": cv_err_lst,
                "lowest_cv_error": lowest_cv_error,
                "best_params": best_params,
                "param_lst": param_lst,
                "testing_error": testing_error,
                "best_bag_k_error_series": old_bag_k_error_series,
            }
            res_log_npm["log"].append(reslog_one_trial)
            print(
                "Trial: ",
                trial_id,
                " Best params: ",
                best_params,
                " Lowest Error: ",
                lowest_cv_error,
                " Testing Error: ",
                testing_error,
            )
            res_log_npm = dump_single_res(res_log_npm, filename)
        res_log_npm["noise_level_lowest_cv_MSE"].append(
            np.mean(trials_loweset_cv_MSE_temp)
        )
        res_log_npm["trials_testing_score"].append(np.mean(trials_testing_score_temp))
        res_log_npm = dump_single_res(res_log_npm, filename)






if __name__ == "__main__":
    args = get_parser().parse_args()
    configs = None
    with open(os.path.join(args.config_path, args.config_name), "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    out_path = args.output_path
    try:
        os.makedirs(out_path)
    except TypeError:
        out_path = "./memory"
    if not os.path.exists(out_path):
        os.makedirs(out_path)


    n_tmp = configs["TEST"]["n"]
    p_tmp = configs["TEST"]["p"]
    m_tmp = configs["TEST"]["m"]
    noise_level_lst = configs["TEST"]["noise_levels"]
    model_name = configs["TEST"]["model"]
    cv_num = configs["TEST"]["cv_num"]
    trial_num = configs["TEST"]["trial_num"]

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
    fixed_params, param_grid = get_model_params(configs)
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    filename = "Baseline_"+configs["filename"].split(".")[0] + "_" + timestamp + ".pkl"
    filename = os.path.join(out_path, filename)
    for n, p, m in npm_lst:
        reslog_npm = run_trials_npm_multi_noise_lvl(
            n,
            p,
            m,
            noise_level_lst,
            model_name,
            fixed_params,
            param_grid,
            cv_num,
            trial_num,
            filename
        )

    print("Done!")
    print("Results are saved in: ", filename)
