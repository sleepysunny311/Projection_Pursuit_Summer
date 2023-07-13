import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pickle as pkl
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from itertools import product
from datetime import datetime
from algorithms import BOMP
from data_generation import *


import warnings

warnings.filterwarnings("ignore")


def get_model_params(config):
    all_params = OmegaConf.to_container(config, resolve=True)["MODEL"]
    param_grid = {}
    fixed_params = {}

    Bag_lst = all_params["Bag_lst"]
    K_lst = all_params["K_lst"]

    del all_params["Bag_lst"]
    del all_params["K_lst"]

    for param, value in all_params.items():
        if isinstance(value, list):
            param_grid[param] = value
        else:
            fixed_params[param] = value

    fixed_params["Bag_lst"] = Bag_lst
    fixed_params["K_lst"] = K_lst
    return fixed_params, param_grid


def run_trials_npm_multi_noise_lvl(
    n, p, m, noise_level_lst, model_name, fixed_params, param_grid, cv_num, trial_num
):
    # get the model

    if model_name == "BOMP":
        model = BOMP(**fixed_params)

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
        "noise_level_lowest_MSE": [],
        "log": [],
    }
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
            best_estimator.set_Bag_lst([best_estimator.optimal_bag])
            best_estimator.set_K_lst([best_estimator.optimal_k])
            best_estimator.fit(X_train, y_train)
            testing_error = mean_squared_error(y_test, best_estimator.predict(X_test))
            trials_testing_score_temp.append(testing_error)
            lowest_cv_error = np.min(cv_err_lst)
            trials_loweset_cv_MSE_temp.append(lowest_cv_error)
            best_params = gs.best_params_
            reslog_one_trial = {
                "noise_level": noise_level,
                "trial": trial_id,
                "cv_error_lst": cv_err_lst,
                "lowest_cv_error": lowest_cv_error,
                "best_params": best_params,
                "best_bag": best_estimator.optimal_bag,
                "best_k": best_estimator.optimal_k,
                "param_lst": param_lst,
                "testing_error": testing_error,
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
        res_log_npm["noise_level_lowest_cv_MSE"].append(
            np.mean(trials_loweset_cv_MSE_temp)
        )
        res_log_npm["trials_testing_score"].append(np.mean(trials_testing_score_temp))
        print(
            "Noise level: ",
            noise_level,
            " Avg Testing Lowest MSE: ",
            np.mean(trials_testing_score_temp),
        )
    return res_log_npm


@hydra.main(config_path="configs", config_name="bomp_default.yaml")
def main(configs: DictConfig):
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
    timestamp = datetime.now().strftime("m%d-%H%M%S")
    filename = configs["filename"].split(".")[0] + "_" + timestamp + ".pkl"
    for n, p, m in npm_lst:
        ALL_LOGS = None
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
        )
        ALL_LOGS = None
        try:
            with open(filename, "rb") as f:
                ALL_LOGS = pkl.load(f)
        except:
            ALL_LOGS = []

        ALL_LOGS.append(reslog_npm)
        with open(filename, "wb") as f:
            pkl.dump(ALL_LOGS, f)

    print("Done!")
    print("Results are saved in: ", filename)
    return ALL_LOGS


if __name__ == "__main__":
    main()
