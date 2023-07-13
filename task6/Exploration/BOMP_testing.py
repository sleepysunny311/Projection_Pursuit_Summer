import hydra
from omegaconf import DictConfig
import numpy as np
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from itertools import product

from algorithms import BOMP
from data_generation import *


import warnings
warnings.filterwarnings('ignore')





    

def get_model_params(config):
    all_params = config['MODEL']
    param_grid = {}
    fixed_params = {}

    K_lst = all_params['K_lst']
    del all_params['K_lst']

    for param, value in all_params.items():
        if isinstance(value, list):
            param_grid[param] = value
        else:
            fixed_params[param] = value
    
    fixed_params['K_lst'] = K_lst
    return fixed_params, param_grid
    
def run_trials_npm_multi_noise_lvl(n, p, m, noise_level_lst, model_name, fixed_params, param_grid, cv_num, trial_num):
    # get the model

    if model_name == "BOMP": 
        model = BOMP(**fixed_params)

    res_log_npm = {
        'parameters': {'n': n, 'p': p, 'm': m, 'noise_level_lst': noise_level_lst, 'model_name': model_name, 'cv_num': cv_num, 'trial_num': trial_num, 'param_grid': param_grid, 'fixed_params': fixed_params},
        'noise_level_lowest_MSE': [],
        'log': []
    }
    print(f"Running trials for n = {n}, p = {p}, m = {m}")
    for noise_level in noise_level_lst:
        print("Cross validating alpha under noise level: ", noise_level)
        trials_loweset_MSE_temp = []
        for trial_id in range(trial_num):
            Data_Geneartor = GaussianDataGenerator(p, n, m, noise_level, trial_id)
            true_signal, dictionary, true_indices, true_coefficients, perturbed_signal = Data_Geneartor.shuffle()
            gs = GridSearchCV(model, param_grid, cv=cv_num, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
            gs.fit(dictionary, perturbed_signal)
            cv_err_lst = -gs.cv_results_['mean_test_score']
            param_lst = gs.cv_results_['params']
            lowest_error = np.min(cv_err_lst)
            trials_loweset_MSE_temp.append(lowest_error)
            best_params = gs.best_params_
            reslog_one_trial = {'noise_level': noise_level, 'trial': trial_id, 'cv_error_lst': cv_err_lst, 
                            'lowest_error': lowest_error, 'best_params': best_params, 'param_lst': param_lst}
            res_log_npm['log'].append(reslog_one_trial)
            print("Trial: ", trial_id, " Best params: ", best_params, " Lowest Error: ", lowest_error)
        res_log_npm['noise_level_lowest_MSE'].append(np.mean(trials_loweset_MSE_temp))
        print("Noise level: ", noise_level, " Avg Lowest MSE: ", np.mean(trials_loweset_MSE_temp))
    return res_log_npm

def run_tests(config, output_path):
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
    
    
    for n, p, m in npm_lst:
        ALL_LOGS = None
        reslog_npm = run_trials_npm_multi_noise_lvl(n, p, m, noise_level_lst, model_name, fixed_params, param_grid, cv_num, trial_num)
        ALL_LOGS = None
        try:
            with open(output_path, 'rb') as f:
                ALL_LOGS = pkl.load(f)
        except:
            ALL_LOGS = []

        ALL_LOGS.append(reslog_npm)
        with open(output_path, 'wb') as f:
            pkl.dump(ALL_LOGS, f)

    print("Done!")
    print("Results are saved in: ", output_path)
    return ALL_LOGS

@hydra.main(config_path='config', config_name='bomp_default.yaml')
def main():

    





if __name__ == '__main__':

    main()
    

        
