import argparse
import glob
import multiprocessing as mp
import os
import time
import yaml
import pickle as pkl

def get_parser():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config-file', type=str, default='configs/defualt_config.yaml', metavar= "FILE" ,help='path to config file')
    parser.add_argument("--output", type=str, help="Output path")
    return parser


def get_cfg(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_cfg(default_config, input_config):
    for key in default_config.keys():
        if isinstance(default_config[key], dict):
            for subkey in default_config[key].keys():
                if subkey in input_config and input_config[subkey] is not None:
                    default_config[key][subkey] = input_config[subkey]
        elif key in input_config and input_config[key] is not None:
            default_config[key] = input_config[key]
    return default_config
    
    
def get_output_path(output_path, config_filename):
    if output_path is None:
        # output file will be a pickle file in the outputs folder
        output_path = os.path.join("outputs", config_filename.split("/")[-1].split(".")[0] + ".pkl")
    else:
        # output file will be a pickle file in the specified folder
        output_path = os.path.join(output_path, config_filename.split("/")[-1].split(".")[0] + ".pkl")
    return output_path


def run_test(config):
    print(config)
    return

def run_one_trial():
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
        
    
    
    
    
