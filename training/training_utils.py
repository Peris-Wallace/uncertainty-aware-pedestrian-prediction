from pathlib import Path
import numpy as np
import torch
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from training.dataset import prepare_data, PIECrossingDataset
from training.model import PedestrianCrossingTransformer
from utils import TrainingTimeCallback, flatten_config




def create_datasets(config):

    prepared = prepare_data(dataset="PIE", configs=config)

    feature_opts = config.get("feature_opts", {})
    expected_input_dim = config["net_opts"]["input_dimension"]
   
    dataset_args = {
        "feature_opts": feature_opts,
        "expected_input_dim": expected_input_dim,
        "dtype": torch.float64,
    }

    return {
        "train": PIECrossingDataset(prepared.train_data, **dataset_args),
        "val": PIECrossingDataset(prepared.val_data, **dataset_args),
        "test": PIECrossingDataset(prepared.test_data, **dataset_args),
    }


def create_dataloaders(config, datasets, seed):
    """Create DataLoaders for train, validation and test splits.
    Only the training loader is shuffled. Validation and test order are kept
    fixed so repeated evaluation uses the same sample ordering. """

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]
    
    train_opts = config['train_opts']
    num_workers = train_opts.get('num_workers', 8)
    persistent_workers = train_opts.get('persistent_workers', True) and num_workers > 0

    common_options = {
        'batch_size': train_opts['batch_size'],
        'num_workers': num_workers,
        'pin_memory': train_opts.get('pin_memory', True),
        'persistent_workers': persistent_workers,
    }

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_options,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_options,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def get_binary_class_distribution(dataset):
    # Return class-distribution statistics for a dataset.

    if hasattr(dataset, "targets"):
        labels = torch.as_tensor(dataset.targets).reshape(-1).long()
    elif hasattr(dataset, "labels"):
        labels = torch.as_tensor(dataset.labels).reshape(-1).long()
    else:
        labels = torch.tensor(
            [int(dataset[index][-1]) for index in range(len(dataset))],
            dtype=torch.long,
        )

    total = int(labels.numel())
    non_crossing = int((labels == 0).sum().item())
    crossing = int((labels == 1).sum().item())

    if total == 0:
        crossing_percentage = 0.0
        non_crossing_percentage = 0.0
        majority_accuracy = 0.0
        imbalance_ratio = float("nan")

    else:
        crossing_percentage = 100.0 * crossing / total
        non_crossing_percentage = 100.0 * non_crossing / total
        majority_accuracy = max(crossing, non_crossing) / total
        minority_count = min(crossing, non_crossing)
        majority_count = max(crossing, non_crossing)
        
        # Ratio > 1 indicates imbalance. A perfectly balanced split gives 1:1.
        imbalance_ratio = (
            majority_count / minority_count
            if minority_count > 0
            else float("inf")
        )

    return {
        "total": total,
        "non_crossing": non_crossing,
        "crossing": crossing,
        "non_crossing_percentage": non_crossing_percentage,
        "crossing_percentage": crossing_percentage,
        "majority_accuracy": majority_accuracy,
        "imbalance_ratio": imbalance_ratio,
    }


def print_class_distribution_summary(datasets):
    # Print class distributions for train, validation and test sets.

    print("\nClass distribution")
    print("-" * 100)
    print(
        f"{'Split':<12}"
        f"{'Total':>10}"
        f"{'Non-cross':>14}"
        f"{'Cross':>10}"
        f"{'Cross %':>12}"
        f"{'Ratio':>12}"
    )
    print("-" * 100)

    distributions = {}
    split_names = {"train": "Train", "val": "Validation", "test": "Test"}
    
    for split_key, display_name in split_names.items():
        stats = get_binary_class_distribution(datasets[split_key])
        distributions[split_key] = stats

        ratio_text = (
            f"{stats['imbalance_ratio']:.2f}:1"
            if np.isfinite(stats["imbalance_ratio"])
            else "inf"
        )

        print(
            f"{display_name:<12}"
            f"{stats['total']:>10}"
            f"{stats['non_crossing']:>14}"
            f"{stats['crossing']:>10}"
            f"{stats['crossing_percentage']:>11.2f}%"
            f"{ratio_text:>12}"
        )

    print("-" * 100)
    return distributions


def verify_dataset(config, datasets, dataloaders):
    # verify the dataset and print information about the samples, 
    # including shapes and dtypes expected by the model
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    expected_sequence_length = (
        config['model_opts']['obs_length'] - 
        (1 if config['model_opts'].get('normalize_boxes', False) else 0)
    )

    expected_input_dimension = config["net_opts"]["input_dimension"]
    
    # Inspect one training sample
    features, target = train_dataset[0]

    print('\nDataset verification')
    print('--------------------')

    print("Train_dataset type:", type(train_dataset))
    print("Train_dataset repr:", repr(train_dataset))
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Train batches: {len(dataloaders['train'])}")
    print("Features type:", type(features))
    print("Features shape:", features.shape)
    print("Target type:", type(target))
    print("Target shape:", target.shape)

    # Each sample must be a two-dimensional sequence:
    # [sequence_length, feature_dimension].
    if features.ndim != 2:
        raise ValueError(
            "Expected sequence features with shape "
            "[sequence_length, input_dimension], "
            f"received {tuple(features.shape)}."
        )

    if features.shape[0] != expected_sequence_length:
        raise ValueError(
            f"Expected observation length {expected_sequence_length}, "
            f"received {features.shape[0]}."
        )

    if features.shape[-1] != expected_input_dimension:
        raise ValueError(
            f"Expected input dimension {expected_input_dimension}, "
            f"received {features.shape[-1]}."
        )

    # Print class distribution summary for train, validation and test sets.
    class_distributions = print_class_distribution_summary(datasets)

    return class_distributions



def build_model(config):
    # Construct the TabularTransformer model based on the configuration.
    model_opts = config["model_opts"]
    net_opts = config["net_opts"]
    obs_length = model_opts["obs_length"]

    # The current box-normalisation pipeline removes one timestep.
    seq_len = (
        obs_length - 1
        if model_opts.get("normalize_boxes", False)
        else obs_length
    )

    model = PedestrianCrossingTransformer(
        input_dim=net_opts["input_dimension"],
        seq_len=seq_len,
        d_model=net_opts.get("d_model", 8),
        n_heads=net_opts.get("num_attention_heads", 2),
        ff_dim=net_opts.get("ff_dim", 16),
        num_layers=net_opts.get("num_layers", 2),
        dropout=net_opts.get("dropout", 0.1),
    )

    return model.to(dtype=torch.float64)


def create_callbacks(config):
    # Create early stopping, model checkpoint and timing callbacks.
    early = config.get('early_stopping', {})
    checkpoint = config['checkpoint']

    checkpoint_directory = Path(checkpoint['directory']).expanduser()
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Add early stopping when enabled to stop training 
    #if the validation metric does not improve for a number of epochs.
    early_stopping_enabled = bool(early.get('enabled', True))
    
    if early_stopping_enabled:
        early_stopping_callback = (
            pl.callbacks.EarlyStopping(
                monitor=early['monitor'],
                min_delta=early['min_delta'],
                patience=early['patience'],
                verbose=True,
                mode=early['mode'],
            )
        )

        callbacks.append(early_stopping_callback)

        print(
            f"Early stopping enabled: monitor={early['monitor']}, "
            f"min_delta={early['min_delta']}, patience={early['patience']}, "
        )
    else:
        early_stopping_callback = None
        print("Early stopping disabled.")

    # Save only the best checkpoint according to the configured validation metric
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(checkpoint_directory),
        monitor=checkpoint['monitor'],
        mode=checkpoint['mode'],
        save_top_k=checkpoint['save_top_k'],
        save_last=False,
        filename=checkpoint['filename'],
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    callbacks.append(checkpoint_callback)
    # Total training duration
    callbacks.append(TrainingTimeCallback())

    return callbacks

def create_mlflow_logger(config):
    # Create an MLFlow logger for experiment tracking and logging.
    experiment = config['experiment']
    mlflow_options = config['mlflow']

    logger = MLFlowLogger(
        experiment_name=experiment['name'],
        run_name=experiment['run_name'],
        tracking_uri=mlflow_options['tracking_uri'],
        tags=experiment.get('tags', {}),
        log_model=mlflow_options.get('log_model', True),
    )

    return logger


def log_experiment_configuration(config, logger, base_config_path, 
                        experiment_config_path, datasets, dataloaders):
    # Log the experiment configuration and dataset statistics to MLFlow.
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    parameters = flatten_config(config)
    parameters.update(
        {
            'config.base_file':
                str(Path(base_config_path).expanduser().resolve()),
            'config.experiment_file':
                str(Path(experiment_config_path).expanduser().resolve()),
            'data.train_samples': len(train_dataset),
            'data.validation_samples': len(val_dataset),
            'data.test_samples': len(test_dataset),
            'data.train_batches': len(dataloaders['train']),
        }
    )

    logger.log_hyperparams(parameters)


def create_trainer(config, logger, callbacks):
    # Construct the PyTorch Lightning Trainer from the YAML configuration.
    train_opts = config['train_opts']

    return pl.Trainer(
        accelerator=train_opts.get('accelerator', 'cuda'),
        devices=train_opts.get('devices', 1),
        max_epochs=train_opts['epochs'],
        precision=train_opts.get('precision', '32-true'),
        callbacks=callbacks,
        logger=logger,
        deterministic=train_opts.get('deterministic', True),
        log_every_n_steps=train_opts.get('log_every_n_steps', 10),
    )

def update_uncertainty_statistics(module, uncertainty, stage):
    # Update the running sum and count of uncertainty values for a given stage.
    uncertainty = uncertainty.detach().reshape(-1)

    getattr(module, f"{stage}_uncertainty_sum").add_(uncertainty.sum())
    getattr(module, f"{stage}_uncertainty_count").add_(uncertainty.numel())


def log_uncertainty_statistics(module, stage):
    # Compute, log and reset the mean evidential uncertainty for an epoch.
    total_count = getattr(module, f"{stage}_uncertainty_count")

    mean_uncertainty = (
        getattr(module, f"{stage}_uncertainty_sum")
        / total_count.clamp_min(1)
    )

    module.log(
        f"{stage}_mean_uncertainty",
        mean_uncertainty,
        on_step=False,
        on_epoch=True,
        prog_bar=False,
        logger=True,
    )

    # Reset accumulators so the next epoch starts from zero.
    getattr(module, f"{stage}_uncertainty_sum").zero_()
    getattr(module, f"{stage}_uncertainty_count").zero_()
    
    return {"mean": mean_uncertainty}
    

def save_model_summary(model, dataset, output_directory):
    """Print the torchinfo model summary and save it as a text file.
    The summary records input/output shapes and trainable parameter counts"""

    output_directory = Path(output_directory).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    features, _ = dataset[0]
    inputs = features.unsqueeze(0).to(dtype=torch.float64)
    
    model_summary = summary(
        model,
        input_data=inputs,
        device="cpu",
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "trainable",
        ),
        depth=5,
        verbose=0,
    )

    summary_path = output_directory / "model_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("MODEL ARCHITECTURE\n")
        file.write("==================\n\n")
        file.write(str(model))
        file.write("\n\n")
        file.write("SUMMARY\n")
        file.write("=================\n\n")
        file.write(str(model_summary))
        file.write("\n")

    print(f"\nSaved model summary: {summary_path}")
    return summary_path