import argparse
import glob
import multiprocessing as mp
import os
import time
import yaml
import pickle as pkl


from data_generation import generate_gaussian_noises_dict, generate_sparse_response, generate_perturbed_response

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


def run_test(config):
    model_params = config['MODEL']
    
    print(config)
    return

def run_one_trial():
    
    # Check if the trial has been done
    
    # Run the trial for the given parameters
    
    # Dump the results to the output path
    
    return

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    # Merge default config and input config
    default_config = get_cfg("configs/default_config.yaml")
    input_config = get_cfg(args.config_file)
    config = merge_cfg(default_config, input_config)
    
    # Output path
    output_path = get_output_path(args.output, args.config_file)
    
    # res = run_test(config)
    # with open(output_path, 'wb') as f:
    #     pkl.dump(res, f)
        
    
    
    
    
