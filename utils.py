import argparse
import time
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
import yaml


def get_device():
    return torch.device("cuda:1" if torch.cuda.is_available() else "cpu" )

def one_hot_embedding(labels, num_classes=2):
    return F.one_hot(
        labels.long(),
        num_classes=num_classes,
    ).float()


def relu_evidence(y):
    return F.relu(y)

# Command-line arguments

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/base.yaml",
        help="Path to the shared base configuration.",
    )

    parser.add_argument(
        "--experiment-config",
        type=str,
        default="configs/experiments/baseline.yaml",
        help="Path to the experiment override configuration.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume checkpoint .ckpt path.",
    )

    return parser.parse_args()


# Configuration helpers

def load_yaml(path):
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must be a YAML mapping: {path}"
        )

    return config


def merge_configs(base, experiment):
    # Merges an experiment configuration into a base configuration. 
    # Values in experiment take priority.
    
    merged = deepcopy(base)

    for key, exp_value in experiment.items():
        base_value = merged.get(key)

        if (isinstance(base_value, dict) and isinstance(exp_value, dict)):
            merged[key] = merge_configs(base_value, exp_value)
        else:
            merged[key] = deepcopy(exp_value)

    return merged


def load_experiment_config(base_path, experiment_path):
    base_config = load_yaml(base_path)
    experiment_config = load_yaml(experiment_path)

    return merge_configs(base_config, experiment_config)


def flatten_config(config, parent_key="", separator="."):

    # Convert nested YAML values into flat parameters.
    flattened = {}

    for key, value in config.items():
        full_key = (
            f"{parent_key}{separator}{key}"
            if parent_key
            else str(key)
        )

        if isinstance(value, dict):
            flattened.update(
                flatten_config(
                    value,
                    parent_key=full_key,
                    separator=separator,
                )
            )

        elif isinstance(value, (list, tuple)):
            flattened[full_key] = ",".join(
                str(item) for item in value
            )

        elif value is None:
            flattened[full_key] = "None"

        else:
            flattened[full_key] = value

    return flattened


def save_resolved_config(config, output_directory):

    output_directory = Path(output_directory).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (output_directory/ 'config.yaml')

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return output_path

def resolve_resume_checkpoint(command_line_checkpoint, resume_config=None):
    # Checkpoint resume path resolution.
    if command_line_checkpoint:
        checkpoint_path = (
            command_line_checkpoint
        )

    elif (
        resume_config
        and resume_config.get(
            "enabled",
            False,
        )
    ):
        checkpoint_path = resume_config.get(
            "checkpoint_path"
        )

    else:
        return None

    if not checkpoint_path:
        raise ValueError('Resume is enabled, but checkpoint_path is empty.')

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Resume checkpoint not found: "
            f"{checkpoint_path}"
        )

    if checkpoint_path.suffix != ".ckpt":
        raise ValueError(
            "Expected a Lightning .ckpt file: "
            f"{checkpoint_path}"
        )

    return str(checkpoint_path)


# Training time callback

class TrainingTimeCallback(pl.Callback):

    def __init__(self):
        super().__init__()

        self.fit_start_time = None
        self.epoch_start_time = None
        self.total_training_seconds = 0.0 

    def on_fit_start(self, trainer, pl_module):
        self.fit_start_time = time.perf_counter()
        self.total_training_seconds = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_start_time is None:
            return

        epoch_seconds = time.perf_counter() - self.epoch_start_time

        self.total_training_seconds += epoch_seconds
        self.epoch_start_time = None

    def on_fit_end(self, trainer, pl_module):
        if self.fit_start_time is None:
            return

        total_time = (time.perf_counter() - self.fit_start_time) / 60.0
        training_time = self.total_training_seconds / 60.0

        print('\nTraining Complete! :)')
        print(f'\nTotal training time: {training_time:.2f} minutes')
        print(f'Total time taken: {total_time:.2f} minutes')
        
        if trainer.logger is not None:
            trainer.logger.log_metrics({
                "total_time": total_time,
                'training_time': training_time
            },
                step=trainer.current_epoch,
            )
            