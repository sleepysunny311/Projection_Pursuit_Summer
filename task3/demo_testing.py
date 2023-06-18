import numpy as np
import pandas as pd
import pickle as pkl
import os
from algorithms import OMP
from data_generation import GaussianDataGenerator

import warnings
warnings.filterwarnings('ignore')


def cv_split(true_signal, dictionary, cv_num):
    true_signal = true_signal.ravel()
    # true_signal is (1200, 1) and dictionary is (1200, 10000), cv both signal and dictionary by rows
    cv_signal = np.split(true_signal, cv_num)
    cv_dictionary = np.split(dictionary, cv_num)
    # Get the list of train and test set
    cv_res = []
    for i in range(cv_num):
        train_signal = np.concatenate(cv_signal[:i] + cv_signal[i + 1:], axis = 0)
        train_dictionary = np.concatenate(cv_dictionary[:i] + cv_dictionary[i + 1:], axis=0)
        test_signal = cv_signal[i]
        test_dictionary = cv_dictionary[i]
        cv_res.append((train_signal, train_dictionary, test_signal, test_dictionary))
    return cv_res

def cal_cv_error(algorithm, cv_num, signal, dictionary):
    cv_res = cv_split(signal, dictionary, cv_num)
    error_lst = []
    for i in range(cv_num):
        train_signal, train_dictionary, test_signal, test_dictionary = cv_res[i]
        algorithm.fit(train_signal, train_dictionary)
        error_lst.append(algorithm.score(test_signal, test_dictionary))
    return np.mean(error_lst)

def cv_best_K(signal, dictionary, cv_num, K_lst):
    K_cv_error = []
    for K in K_lst:
        OMP_tmp = OMP(K, ignore_warning=True)
        K_cv_error.append(cal_cv_error(OMP_tmp, cv_num, signal, dictionary))
    lowest_error = np.min(K_cv_error)
    lowest_error_K = K_lst[np.argmin(K_cv_error)]
    return lowest_error, lowest_error_K, K_cv_error


# Improvement: Save the result to a file

if not os.path.exists('./memory'):
    os.mkdir('./memory')

def cv_best_K_noise_level_multi_trial(N, d, m, noise_level_lst, cv_num, K_lst, trial_num, output_filename = None):
    if output_filename is None:
        output_filename = str(N) + '_' + str(d) + '_' + str(m) + '_' + str(trial_num) + '_' + str(cv_num) + '.pkl'
    res_log = {
        'parameters': {'N': N, 'd': d, 'm': m, 'noise_level_lst': noise_level_lst, 'cv_num': cv_num, 'trial_num': trial_num, 'K_lst': K_lst},
        'noise_level_best_K': [],
        'noise_level_lowest_MSE': [],
        'log': []
    }
    noise_level_best_K = []
    noise_level_lowest_MSE = []
    for noise_level in noise_level_lst:
        print("Cross validating K under noise level: ", noise_level)
        trials_best_K_tmp = []
        MSE_loweset_K_temp = []
        for trial in range(trial_num):
            Data_Geneartor = GaussianDataGenerator(N, d, m, noise_level, trial)
            true_signal, dictionary, true_indices, true_coefficients, perturbed_signal = Data_Geneartor.shuffle()
            lowest_error, lowest_error_K, cv_err_lst = cv_best_K(perturbed_signal, dictionary, cv_num, K_lst)
            trials_best_K_tmp.append(lowest_error_K)
            MSE_loweset_K_temp.append(lowest_error)
            print("Trial: ", trial, " Best K: ", lowest_error_K, " Lowest Error: ", lowest_error)
            log_tmp = {'noise_level': noise_level, 'trial': trial, 'data': Data_Geneartor, 'cv_error_lst': cv_err_lst, 
                       'lowest_error': lowest_error, 'lowest_error_K': lowest_error_K}
            res_log['log'].append(log_tmp)
        noise_level_best_K.append(np.mean(trials_best_K_tmp))
        noise_level_lowest_MSE.append(np.mean(MSE_loweset_K_temp))
        print("Average best K for noise level: ", noise_level, " is: ", np.mean(trials_best_K_tmp), " with MSE: ", np.mean(MSE_loweset_K_temp))
    res_log['noise_level_best_K'] = noise_level_best_K
    res_log['noise_level_lowest_MSE'] = noise_level_lowest_MSE
    with open('./memory/' + output_filename, 'wb') as f:
        pkl.dump(res_log, f)
    print("Finished!")
    print("Log file saved to: ", './memory/' + output_filename)
    return noise_level_best_K, noise_level_lowest_MSE, res_log


N = 1000
d = 1200
m = 20
noise_level_lst = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
trial_num = 5

cv_num = 5
# ? Is cv necessary? 

noise_level_best_K, noise_level_lowest_MSE, res_log = cv_best_K_noise_level_multi_trial(N, d, m, noise_level_lst, cv_num, np.arange(1, 41, 1), trial_num, output_filename = None)