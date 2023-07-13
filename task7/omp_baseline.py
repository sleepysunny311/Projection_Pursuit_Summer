import numpy as np
import pandas as pd
import pickle as pkl
import os
from data_generation import *
from OMP import OMP
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

N = 1000
d = 400
m = 40
noise_level_lst = [round(x, 2) for x in np.arange(0, 0.52, 0.02)]
trial_num = 10
cv_num = 5


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
    noise_level_testing_score = []
    for noise_level in noise_level_lst:
        print("Cross validating K under noise level: ", noise_level)
        trials_best_K_tmp = []
        MSE_loweset_K_temp = []
        testing_score_tmp = []
        for trial in range(trial_num):
            Data_Geneartor = GaussianDataGenerator(N, d, m, noise_level, trial)
            true_signal, dictionary, true_indices, true_coefficients, perturbed_signal = Data_Geneartor.shuffle()
            X_train, X_test, y_train, y_test = train_test_split(dictionary, perturbed_signal, test_size=0.2, random_state=trial)
            lowest_error, lowest_error_K, cv_err_lst = cv_best_K(y_train, X_train, cv_num, K_lst)
            best_estimator = OMP(lowest_error_K, ignore_warning=True)
            best_estimator.fit(y_train, X_train)
            testing_score = best_estimator.score(y_test, X_test)
            testing_score_tmp.append(testing_score)
            trials_best_K_tmp.append(lowest_error_K)
            MSE_loweset_K_temp.append(lowest_error)
            print("Trial: ", trial, " Best K: ", lowest_error_K, " Lowest Error: ", lowest_error, " Testing Score: ", testing_score)
            log_tmp = {'noise_level': noise_level, 'trial': trial, 'cv_error_lst': cv_err_lst, 
                       'lowest_error': lowest_error, 'lowest_error_K': lowest_error_K, 'testing_score': testing_score}
            res_log['log'].append(log_tmp)
        noise_level_best_K.append(np.mean(trials_best_K_tmp))
        noise_level_lowest_MSE.append(np.mean(MSE_loweset_K_temp))
        noise_level_testing_score.append(np.mean(testing_score_tmp))
        print("Average best K for noise level: ", noise_level, " is: ", np.mean(trials_best_K_tmp), " with CV Error: ", np.mean(MSE_loweset_K_temp), " and testing score: ", np.mean(testing_score_tmp))
    res_log['noise_level_best_K'] = noise_level_best_K
    res_log['noise_level_lowest_MSE'] = noise_level_lowest_MSE
    with open('./memory/' + output_filename, 'wb') as f:
        pkl.dump(res_log, f)
    print("Finished!")
    print("Log file saved to: ", './memory/' + output_filename)
    return noise_level_best_K, noise_level_lowest_MSE, res_log


noise_level_lst = [round(x, 2) for x in np.arange(0, 0.52, 0.02)]
N = 1000
d = 600
m = 20
trial_num = 20
cv_num = 5
K_lst = np.arange(1, m+20+1, 1).tolist()

if __name__ == "__main__":
    (
        noise_level_best_K,
        noise_level_lowest_MSE,
        res_log,
    ) = cv_best_K_noise_level_multi_trial(
        N,
        d,
        m,
        noise_level_lst,
        cv_num,
        K_lst,
        trial_num,
        output_filename="omp_baseline_testing_full.pkl",
    )
