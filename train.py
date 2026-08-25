"""
Training entry point for the pedestrian crossing prediction model.

This script performs the complete experiment workflow:

1. Parse command-line arguments.
2. Load and merge the base and experiment configurations.
3. Configure reproducibility.
4. Prepare datasets and dataloaders.
5. Build the pedestrian crossing model.
6. Save and log the resolved configuration and model summary.
7. Configure MLflow, callbacks, and the Lightning Trainer.
8. Train or resume the model.
9. Test the best saved checkpoint.

Example
-------
Run an experiment from the project root:

    python -m training.train \
        --base-config configs/base.yaml \
        --experiment-config configs/experiments/baseline.yaml

Resume from a checkpoint:

    python -m training.train \
        --base-config configs/base.yaml \
        --experiment-config configs/experiments/baseline.yaml \
        --resume checkpoints/trep_baseline/best.ckpt
"""
# Import libraries
import os  

os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG",
    ":4096:8",
)

import torch
import pytorch_lightning as pl

# Import custom modules 
from training.lightning_model import PedestrianCrossingLightningModule
from training.training_utils import (
    build_model,
    create_callbacks,
    create_dataloaders,
    create_datasets,
    create_mlflow_logger,
    create_trainer,
    log_experiment_configuration,
    verify_dataset,
    save_model_summary
)

from utils import load_experiment_config, parse_args, resolve_resume_checkpoint, save_resolved_config


def main():
    # Runs one complete training, validation, and testing experiment.

    args = parse_args()

    # Command-line arguments specify the base configuration, experiment
    # overrides, and an optional checkpoint from which to resume training.
    config = load_experiment_config(base_path=args.base_config,
        experiment_path=(args.experiment_config))


    # Seed Python, NumPy, PyTorch, CUDA, and dataloader workers through PyTorch Lightning for reporducibilty
    seed = config.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # cuDNN benchmarking is enabled only when deterministic execution is disabled.
    deterministic = config["train_opts"].get("deterministic", True)  
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.set_float32_matmul_precision("high")

    # Saves the merged configuration records the exact settings used by
    # this run, including values inherited from the base configuration.
    resolved_config_path = (
        save_resolved_config(
            config=config, 
            output_directory=(config['checkpoint']['directory']),
        )
    )

    # Create and verify the datasets
    datasets = create_datasets(config=config)

    # Construct reproducible training, validation, and test dataloaders.
    dataloaders = create_dataloaders(
        config=config,
        datasets=datasets,
        seed=seed,
    )

    # Check sample counts, sequence length, feature dimension, and dtypes before a training run.
    verify_dataset(
        config=config,
        datasets=datasets,
        dataloaders=dataloaders,
    )

    # Build the underlying model from the resolved architecture parameters in the configuration.
    # and generate the model summary 
    model = build_model(config)
    model_summary_path = save_model_summary(
        model=model,
        dataset=datasets["train"],
        output_directory=config["checkpoint"]["directory"],
    )

    # Wrap the PyTorch model in a LightningModule. This wrapper defines the
    # loss function, optimiser, scheduler, training and validation steps,
    # evaluation metrics, threshold selection, and uncertainty calculations.
    lit_model = PedestrianCrossingLightningModule.from_config(model=model, config=config)
    

    # Initialise MLflow and log experiment artifacts
    logger = create_mlflow_logger(config)

    if config["mlflow"].get("log_artifacts", True):
        logger.experiment.log_artifact(
            logger.run_id,
            str(resolved_config_path),
            artifact_path="config",
        )

        logger.experiment.log_artifact(
            logger.run_id,
            str(model_summary_path),
            artifact_path="model",
        )


    log_experiment_configuration(
        config=config,
        logger=logger,
        base_config_path=(args.base_config),
        experiment_config_path=(args.experiment_config),
        datasets=datasets,
        dataloaders=dataloaders,
    )

    # Configure callbacks and checkpoint resumption

    callbacks = create_callbacks(config)

    resume_checkpoint = (resolve_resume_checkpoint(command_line_checkpoint=(args.resume), resume_config=config.get('resume')))

    if resume_checkpoint:
        print('\nResuming from checkpoint:')
        print(resume_checkpoint)
    else:
        print('\nStarting training from scratch.')


    # Create the Lightning Trainer and fit the model
    trainer = create_trainer(
        config=config,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=resume_checkpoint,
    )

    # Locates the best validation checkpoint
    checkpoint_callback = next(
        callback
        for callback in callbacks
        if isinstance(callback, pl.callbacks.ModelCheckpoint)
    )

    best_validation_f1 = checkpoint_callback.best_model_score
    best_model_path = checkpoint_callback.best_model_path
    
    # Test the model using the best checkpoint
    if best_model_path and best_validation_f1 is not None:
        best_validation_f1 = float(best_validation_f1.detach().cpu().item())
        print("\nBest validation result")
        print("----------------------")
        print(f"Best tuned validation F1: {best_validation_f1:.4f}")
        print(f"Best checkpoint: {best_model_path}")

        # Store the best F1 validation score
        if trainer.logger is not None:
            trainer.logger.log_metrics({'best_validation_f1': best_validation_f1}, step=trainer.current_epoch)

        test_results = trainer.test(
            model=lit_model,
            dataloaders=dataloaders["test"],
            ckpt_path=best_model_path,
        )

        if test_results:
            test_metrics = test_results[0]

            print("\nTEST_METRICS_START")

            for name, value in test_metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()

                print(f"TEST_METRIC {name}={value}")

            print("TEST_METRICS_END\n")
    else:
        # If no checkpoint is available, test the final in-memory model.
        # This may occur when checkpoint saving is conditional on a minimum
        # validation score or when no valid monitored score was produced.
        print(
            "No checkpoint was saved."
            "Testing the final model state instead."
        )

        test_results = trainer.test(
            model=lit_model,
            dataloaders=dataloaders["test"],
            ckpt_path=None,
        )

        if test_results:
            test_metrics = test_results[0]

            print("\nTEST_METRICS_START")

            for name, value in test_metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().cpu().item()

                print(f"TEST_METRIC {name}={value}")

            print("TEST_METRICS_END\n")

if __name__ == "__main__":
    main()