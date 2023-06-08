import numpy as np
import pandas as pd
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor
import os
import hydra
from omegaconf import DictConfig

from algorithms import matching_pursuit, orthogonal_matching_pursuit




def ensure_directory_exists(dir_name):
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except OSError as e:
            print(f"Error creating directory {dir_name}: {e}")
    else:
        pass


def generate_gaussian_noises_dict(N, d, seed=0):
    np.random.seed(seed)
    gaussian_noises = np.random.normal(size=(d, N))
    norms = np.linalg.norm(gaussian_noises, axis=0, keepdims=True)
    # Create unit-norm vectors
    unit_vectors = gaussian_noises / norms
    return unit_vectors

def generate_sparse_response(gaussian_matrix, m, seed=0):
    np.random.seed(seed)
    indices = np.random.choice(gaussian_matrix.shape[1], size=m, replace=False)
    selected_vectors = gaussian_matrix[:, indices]
    coefficients = np.random.normal(size=(m, 1))  # random coefficients for each selected vector
    y = selected_vectors @ coefficients
    return y, indices, coefficients

def generate_perturbed_response(y, noise_level, seed=0):
    np.random.seed(seed)
    norm_y = np.linalg.norm(y)
    noise = np.random.normal(size=y.shape, scale=norm_y * noise_level)
    y_perturbed = y + noise
    return y_perturbed


tested_algorithms_list = [matching_pursuit, orthogonal_matching_pursuit]
N = 100000
d = 300

# TODO: Change the parameters
# true_sparsity_list = [2]
# noise_level_list = [0, 0.01]
# trial_num = 1

true_sparsity_list = [2, 5, 10, 20, 50, 100]
noise_level_list = [0, 0.01, 0.05, 0.1]
trial_num = 20 # number of trials

params = [(true_sparsity, noise_level, algorithm, trial_idx, N, d) for true_sparsity in true_sparsity_list for noise_level in noise_level_list for algorithm in tested_algorithms_list for trial_idx in range(trial_num)]

def run_one_trial(true_sparsity, noise_level, algorithm, trial_idx, N, d):
    # Generate a dictionary
    dictionary = generate_gaussian_noises_dict(N, d, seed=trial_idx)
    # Generate a sparse response
    y, real_indices, real_coefficients = generate_sparse_response(dictionary, true_sparsity, seed=trial_idx)
    # Generate perturbed responses
    y_perturbed = generate_perturbed_response(y, noise_level, seed=trial_idx)
    # Run algorithms
    results = {}
    a, output_indices, output_coefficients = algorithm(y_perturbed, dictionary, true_sparsity)
    results['Algorithm'] = algorithm.__name__
    results['True sparsity'] = true_sparsity
    results['y'] = y
    results['Noise level'] = noise_level
    results['Trial index'] = trial_idx
    results['Real Indices'] = [real_indices]
    results['Real Coefficients'] = [real_coefficients]
    results['Output Indices'] = [output_indices]
    results['Output Coefficients'] = [output_coefficients]
    results['a'] = a
    
    return results

@hydra.main(config_path='./Configs/', config_name="sparse_level_2_5")
def main(configs: DictConfig):
    tested_algorithms = [matching_pursuit, orthogonal_matching_pursuit]
    N = configs.N
    d = configs.d
    
    
    trial_num = configs.trial_num
    noise_level_list = configs.noise_level_list
    true_sparsity_list = configs.true_sparsity_list

    # Feed parameters to run_one_trial
    params = [(true_sparsity, noise_level, tested_algorithms, trial_idx, N, d) for true_sparsity in true_sparsity_list for noise_level in noise_level_list for trial_idx in range(trial_num)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(run_one_trial, *zip(*params))
    results = list(results)
    with open('reports.pkl', 'wb') as f:
        pkl.dump(results, f)
    print("Done!")

if __name__ == '__main__':
    main()
