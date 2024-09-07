import os
import argparse
import yaml
import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', action='store_true', help="Download gpt2-pytorch_model.bin")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'gpu'], help="GPU or CPU acceleration")
    args = parser.parse_args()
    return args


def dict2namespace(config):
    new_config = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(new_config, key, value)
    return new_config


def parse_config():
    with open('config.yml', 'r') as f:
        config = yaml.full_load(f)
    return dict2namespace(config)


