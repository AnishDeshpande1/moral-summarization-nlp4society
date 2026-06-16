import argparse
import itertools
import copy

from .utils import *


def parse_command_line_args(inference=False, training=False):
    parser = argparse.ArgumentParser()

    # Parse basic arguments
    parser.add_argument('--config-file',     type=str,  default="/mnt/moral-summarization/peft_config.yaml")
    parser.add_argument('--hf-model-folder', type=str,  default="/mnt/huggingface-models")
    parser.add_argument('--verbose',         type=bool, action=argparse.BooleanOptionalAction, default=True)

    if inference:
        parser.add_argument('--prompt-dir', type=str, default="/mnt/moral-summarization/results/test_prompts")
        parser.add_argument('--testing',    type=bool, action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--use-ollama', type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--ollama-model', type=str, default="llama3")
        parser.add_argument('--use-chaining', type=bool, action=argparse.BooleanOptionalAction, default=False)

    if training:
        parser.add_argument('--output-dir',   type=str,  default="/mnt/huggingface-models/trained-models")
        parser.add_argument('--dataset-size', type=int,  default=-1)
        parser.add_argument('--bf16',         type=bool, action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument('--fp16',         type=bool, action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument('--lr-scheduler', type=str,  default="linear")
    
    return parser.parse_args()


def read_config_file(config_path):
    return load_yaml(config_path)


def merge_args_in_config(args, config, inference=False, training=False):
    config['verbose'] = args.verbose
    config['hf_model_folder'] = args.hf_model_folder

    if inference:
        if 'inference' not in config:
            config['inference'] = {}
        config['inference']['prompt_dir'] = args.prompt_dir
        config['inference']['testing'] = args.testing
        config['inference']['use_ollama'] = args.use_ollama
        config['inference']['ollama_model'] = args.ollama_model
        config['inference']['use_chaining'] = args.use_chaining
    
    if training:
        config['dataset_size'] = args.dataset_size
        if 'training' not in config:
            config['training'] = {}
        config['training']['output_dir'] = args.output_dir
        config['training']['bf16'] = args.bf16
        config['training']['fp16'] = args.fp16
        config['training']['lr_scheduler_type'] = args.lr_scheduler
    
    return config


def load_config(inference=False, training=False):
    args = parse_command_line_args(inference=inference, training=training)
    config = read_config_file(args.config_file)
    config = merge_args_in_config(args, config, inference=inference, training=training)

    return config


def get_basic_config():
    return {
        'verbose' : True,
        'hf_model_folder' : "/home/eliscio/.cache/huggingface",
        'inference' : {
            'prompt_dir' : "./results/test_prompts",
        }
    }


def get_config_combinations(base_config, hyperparameters):
    keys, values = zip(*hyperparameters.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    configs = []
    for combination in combinations:
        new_config = copy.deepcopy(base_config)
        for key, value in combination.items():
            keys = key.split('.')
            d = new_config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
        configs.append(new_config)

    return configs
