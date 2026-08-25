from pathlib import Path

import torch
from torchinfo import summary
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from training.dataset import prepare_data, tabular_transformer, VideoMAEDataset
from training.model import PedestrianCrossingTransformer, VideoMAEClassifier
from utils import TrainingTimeCallback, flatten_config


def get_model_dtype(config):
    dtype_name = config['net_opts'].get('dtype', 'float64').lower()

    if dtype_name == 'float64': return torch.float64
    if dtype_name == 'float32': return torch.float32

    raise ValueError("net_opts.dtype must be 'float32' or 'float64'.")


def create_datasets(config, dtype=None):

    data_dtype = (
        dtype
        if dtype is not None
        else get_model_dtype(config)
    )

    prepared = prepare_data(
        dataset="PIE",
        configs=config,
    )

    feature_opts = config.get(
        "feature_opts",
        {},
    )

    visual_config = config.get(
        "visual_features",
        {},
    )

    visual_enabled = bool(
        visual_config.get(
            "enabled",
            False,
        )
    )

    visual_encoder = (
        visual_config.get(
            "encoder",
            None,
        )
        if visual_enabled
        else None
    )

    valid_encoders = {
        "resnet18",
        "videomae",
    }

    if (
        visual_enabled
        and visual_encoder
        not in valid_encoders
    ):
        raise ValueError(
            "Visual features are enabled, "
            f"but encoder='{visual_encoder}' "
            f"is invalid. Expected one of "
            f"{valid_encoders}."
        )

    # ---------------------------------------------------------
    # Select encoder-specific visual configuration
    # ---------------------------------------------------------

    if visual_enabled:

        if visual_encoder == "resnet18":
            encoder_config = config.get(
                "resnet_features",
                {},
            )

        elif visual_encoder == "videomae":
            encoder_config = config.get(
                "videomae_features",
                {},
            )

    else:
        encoder_config = {}


    def visual_split(split):
        """
        Return visual feature configuration for one dataset split.

        visual_features.enabled
            controls whether visual information is used.

        visual_features.encoder
            selects the representation type:
            - resnet18
            - videomae
        """

        if not visual_enabled:
            return {
                "enabled": False,
                "encoder": None,
                "feature_dim": 0,
                "feature_file": None,
            }

        file_key = (
            f"{split}_file"
        )

        if file_key not in encoder_config:
            raise KeyError(
                f"Visual encoder '{visual_encoder}' "
                f"is enabled, but "
                f"'{file_key}' is missing from "
                f"its configuration."
            )

        visual_file = Path(
            encoder_config[file_key]
        ).expanduser().resolve()

        if not visual_file.exists():
            raise FileNotFoundError(
                f"Visual feature file for "
                f"split '{split}' does not exist:\n"
                f"{visual_file}"
            )

        feature_dim = int(
            encoder_config.get(
                "feature_dim",
                512
                if visual_encoder == "resnet18"
                else 768,
            )
        )

        return {
            "enabled": True,
            "encoder": visual_encoder,
            "feature_dim": feature_dim,
            "feature_file": str(
                visual_file
            ),
        }


    expected_input_dim = config[
        "net_opts"
    ][
        "input_dimension"
    ]


    # ---------------------------------------------------------
    # VideoMAE is clip-level, not frame-level.
    # ---------------------------------------------------------

    if (
        visual_enabled
        and visual_encoder == "videomae"
    ):

        train_opts = visual_split(
            "train"
        )

        val_opts = visual_split(
            "val"
        )

        test_opts = visual_split(
            "test"
        )

        train_dataset = VideoMAEDataset(
            set_data=prepared.train_data,
            feature_file=train_opts[
                "feature_file"
            ],
            feature_dim=train_opts[
                "feature_dim"
            ],
            dtype=data_dtype,
        )

        val_dataset = VideoMAEDataset(
            set_data=prepared.val_data,
            feature_file=val_opts[
                "feature_file"
            ],
            feature_dim=val_opts[
                "feature_dim"
            ],
            dtype=data_dtype,
        )

        test_dataset = VideoMAEDataset(
            set_data=prepared.test_data,
            feature_file=test_opts[
                "feature_file"
            ],
            feature_dim=test_opts[
                "feature_dim"
            ],
            dtype=data_dtype,
        )

    else:

        # -----------------------------------------------------
        # Existing tabular / ResNet sequence pipeline
        # -----------------------------------------------------

        train_dataset = tabular_transformer(
            prepared.train_data,
            feature_opts=feature_opts,
            expected_input_dim=expected_input_dim,
            visual_opts=visual_split(
                "train"
            ),
            dtype=data_dtype,
        )

        val_dataset = tabular_transformer(
            prepared.val_data,
            feature_opts=feature_opts,
            expected_input_dim=expected_input_dim,
            visual_opts=visual_split(
                "val"
            ),
            dtype=data_dtype,
        )

        test_dataset = tabular_transformer(
            prepared.test_data,
            feature_opts=feature_opts,
            expected_input_dim=expected_input_dim,
            visual_opts=visual_split(
                "test"
            ),
            dtype=data_dtype,
        )


    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def create_dataloaders(config, datasets, seed):

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


def verify_dataset(config, datasets, dataloaders):

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    expected_sequence_length = (
        config['model_opts']['obs_length'] - 
        (1 if config['model_opts'].get('normalize_boxes', False) else 0)
    )

    expected_input_dimension = config['net_opts']['input_dimension']

    print('\nDataset verification')
    print('--------------------')

    print("train_dataset type:", type(train_dataset))
    print("train_dataset repr:", repr(train_dataset))

    sample = train_dataset[0]

    print("sample type:", type(sample))
    print("sample repr:", repr(sample))

    if isinstance(sample, (tuple, list)):
        print("sample length:", len(sample))

        for i, item in enumerate(sample):
            print(
                f"sample[{i}] type:",
                type(item),
                "shape:",
                getattr(item, "shape", None),
            )
    else:
        print(
            "sample shape:",
            getattr(sample, "shape", None),
        )

    features, target = sample

    print('--------------------')
    print(f'Train samples:      {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    print(f'Test samples:       {len(test_dataset)}')
    print(f'Train batches:      {len(dataloaders["train"])}')
    print(f'Feature shape:      {features.shape}')
    print(f'Feature dtype:      {features.dtype}')
    print(f'Target:             {target}')
    print(f'Target dtype:       {target.dtype}')

    visual_config = config.get("visual_features", {})
    visual_enabled = bool(visual_config.get("enabled", False))
    visual_encoder = visual_config.get("encoder", None)

    # VideoMAE:
    # one clip-level embedding
    if visual_enabled and visual_encoder == "videomae":

        if features.ndim != 1:
            raise ValueError(
                "Expected VideoMAE features "
                "with shape "
                f"[{expected_input_dimension}], "
                f"received "
                f"{tuple(features.shape)}."
            )

        if features.shape[0] != expected_input_dimension:
            raise ValueError(
                "Expected VideoMAE input "
                "dimension "
                f"{expected_input_dimension}, "
                f"received "
                f"{features.shape[0]}."
            )

    # Tabular / ResNet:
    # temporal sequence
    else:

        if features.ndim != 2:
            raise ValueError(
                "Expected sequence features "
                "with shape "
                "[sequence_length, "
                "input_dimension], "
                f"received "
                f"{tuple(features.shape)}."
            )

        if features.shape[0] != expected_sequence_length:
            raise ValueError(
                "Expected observation length "
                f"{expected_sequence_length}, "
                f"received "
                f"{features.shape[0]}."
            )

        if (
            features.shape[-1]
            != expected_input_dimension
        ):
            raise ValueError(
                "Expected input dimension "
                f"{expected_input_dimension}, "
                f"received "
                f"{features.shape[-1]}."
            )


def build_model(config):
    model_opts = config["model_opts"]
    net_opts = config["net_opts"]

    visual_config = config.get(
        "visual_features",
        {},
    )

    visual_enabled = bool(
        visual_config.get(
            "enabled",
            False,
        )
    )

    visual_encoder = (
        visual_config.get(
            "encoder",
            None,
        )
        if visual_enabled
        else None
    )

    # ---------------------------------------------------------
    # VideoMAE
    # ---------------------------------------------------------
    #
    # VideoMAE features are already clip-level:
    #
    #     [batch_size, 768]
    #
    # Therefore we use an MLP classifier rather than the
    # temporal Transformer.
    # ---------------------------------------------------------

    if (
        visual_enabled
        and visual_encoder == "videomae"
    ):

        model = VideoMAEClassifier(
            input_dim=net_opts[
                "input_dimension"
            ],
            hidden_dim=net_opts.get(
                "hidden_dimension",
                256,
            ),
            dropout=net_opts.get(
                "dropout",
                0.3,
            ),
            output_dim=net_opts.get(
                "output_dimension",
                2,
            ),
        )

    # ---------------------------------------------------------
    # Tabular / ResNet
    # ---------------------------------------------------------
    #
    # These inputs retain a temporal sequence:
    #
    #     [batch_size, sequence_length, feature_dimension]
    #
    # so they continue to use the Transformer.
    # ---------------------------------------------------------

    else:

        obs_length = model_opts["obs_length"]

        # Box normalisation removes the first timestep because
        # coordinates are represented relative to the first /
        # previous observation.
        if model_opts.get("normalize_boxes", False):
            seq_length = obs_length - 1
        else:
            seq_length = obs_length
            

        model = PedestrianCrossingTransformer(
            ip_dim=net_opts["input_dimension"],
            seq_len=seq_length,
            d_model=net_opts["d_model"],
            nhead=net_opts["num_attention_heads"],
            ff_dim=net_opts["ff_dim"],
            nlayers=net_opts["num_layers"],
            dropout=net_opts["dropout"],
        )

    # Model dtype
    if get_model_dtype(config) == torch.float64:
        model = model.double()
    else:
        model = model.float()
    return model



def create_callbacks(config):
    early = config.get('early_stopping', {})
    checkpoint = config['checkpoint']

    checkpoint_directory = Path(checkpoint['directory']).expanduser()
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Add early stopping when enabled to stop training 
    #if the chosen metric does not improve for a number of epochs.
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

    timing_callback = TrainingTimeCallback()

    callbacks.append(checkpoint_callback)
    callbacks.append(timing_callback)

    return callbacks

def create_mlflow_logger(config):
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


def log_experiment_configuration(config, logger, base_config_path, experiment_config_path, datasets, dataloaders):
    
    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    parameters = flatten_config(config)
    parameters.update(
        {
            'config.base_file':
                str(
                    Path(
                        base_config_path
                    ).expanduser().resolve()
                ),
            'config.experiment_file':
                str(
                    Path(
                        experiment_config_path
                    ).expanduser().resolve()
                ),
            'data.train_samples': len(train_dataset),
            'data.validation_samples': len(val_dataset),
            'data.test_samples': len(test_dataset),
            'data.train_batches': len(dataloaders['train']),
        }
    )

    logger.log_hyperparams(parameters)


def create_trainer(config, logger, callbacks):

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
    uncertainty = uncertainty.detach().reshape(-1)

    getattr(module, f"{stage}_uncertainty_sum").add_(uncertainty.sum())
    getattr(module, f"{stage}_uncertainty_count").add_(uncertainty.numel())


def log_uncertainty_statistics(module, stage):
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

    getattr(module, f"{stage}_uncertainty_sum").zero_()
    getattr(module, f"{stage}_uncertainty_count").zero_()
    
    return {"mean": mean_uncertainty}


def save_model_summary(model, dataset, output_directory):
    # Print the torchinfo model summary and save it as a text file.

    output_directory = (
        Path(output_directory)
        .expanduser()
        .resolve()
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    sample_features, _ = dataset[0]

    model_dtype = next(model.parameters()).dtype

    # Determine model input shape from dataset sample
    if sample_features.ndim == 1:
        # VideoMAE clip-level embedding:
        # [768]
        input_size = (
            1,
            sample_features.shape[0],
        )

    elif sample_features.ndim == 2:
        # Temporal sequence:
        # [sequence_length, input_dimension]
        input_size = (
            1,
            sample_features.shape[0],
            sample_features.shape[1],
        )

    else:
        raise ValueError(
            "Unsupported dataset feature shape for model summary: "
            f"{tuple(sample_features.shape)}"
        )

    model_summary = summary(
        model,
        input_size=input_size,
        dtypes=[model_dtype],
        device="cpu",
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=5,
        verbose=0,
    )

    summary_path = output_directory / "model_summary.txt"
    summary_text = str(model_summary)

    with summary_path.open("w", encoding="utf-8") as file:

        file.write("MODEL ARCHITECTURE\n")
        file.write("==================\n\n")
        file.write(str(model))
        file.write("\n\n")

        file.write("SUMMARY\n")
        file.write("=================\n\n")
        file.write(summary_text)
        file.write("\n")

    print("\nModel summary")
    print("-------------")
    print(model_summary)
    print(f"\nSaved model summary: {summary_path}")

    return summary_path
    