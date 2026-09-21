from __future__ import annotations
import argparse
import shutil
import csv
import math
import re
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml



SEEDS = [
    42,
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1010,
    1111,
    1212,
    1313,
    1414,
    1515
]

# Paths configured from command-line arguments
BASE_CONFIG = None
EXPERIMENT_CONFIG = None
OUTPUT_DIRECTORY = None

TEMP_CONFIG_DIRECTORY = None
LOG_DIRECTORY = None
CHECKPOINT_DIRECTORY = None

RESULTS_CSV = None
SUMMARY_CSV = None


METRIC_PATTERNS = {
    # Classification metrics
    "test_f1": r"TEST_METRIC\s+test_f1=([0-9.eE+-]+)",
    "test_accuracy": r"TEST_METRIC\s+test_accuracy=([0-9.eE+-]+)",
    "test_precision": r"TEST_METRIC\s+test_precision=([0-9.eE+-]+)",
    "test_recall": r"TEST_METRIC\s+test_recall=([0-9.eE+-]+)",
    "test_auroc": r"TEST_METRIC\s+test_auroc=([0-9.eE+-]+)",
    "test_loss": r"TEST_METRIC\s+test_loss=([0-9.eE+-]+)",

    # Calibration metrics
    "test_ece": r"TEST_METRIC\s+test_ece=([0-9.eE+-]+)",
    "test_brier": r"TEST_METRIC\s+test_brier=([0-9.eE+-]+)",

    # Uncertainty metrics
    "test_mean_uncertainty": r"TEST_METRIC\s+test_mean_uncertainty=([0-9.eE+-]+)",
    "test_aurc": r"TEST_METRIC\s+test_aurc=([0-9.eE+-]+)",
}


BASE_REQUIRED_METRICS = [
    "test_f1",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_auroc",
]

CALIBRATION_METRICS = [
    "test_ece",
    "test_brier",
]

UNCERTAINTY_METRICS = [
    "test_mean_uncertainty",
    "test_aurc",
]

# YAML helpers
def load_yaml(path: Path) -> dict[str, Any]:
    # Load a YAML file and verify that it contains a dictionary.
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"{path} does not contain a YAML dictionary.")

    return config


def save_yaml(config, path):
    # Saves a dictionary as YAML
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, default_flow_style=False)


# Create one configuration for each seed

def create_seed_config(original_config, seed):
    # Create a temporary experiment configuration for one seed.
    # The original experiment YAML is not modified.

    config = deepcopy(original_config)

    # Store seed in both common locations.
    config["seed"] = int(seed)

    config.setdefault("train_opts", {})
    config["train_opts"]["seed"] = int(seed)

    # Create a unique experiment run name.
    config.setdefault("experiment", {})
    base_run_name = config["experiment"].get("run_name", "evidential")
    config["experiment"]["run_name"] = f"{base_run_name}-seed-{seed}"

    # Add seed to experiment tags.
    config["experiment"].setdefault("tags", {})
    config["experiment"]["tags"]["seed"] = str(seed)

    # Create a separate checkpoint directory for this seed.
    seed_checkpoint_directory = CHECKPOINT_DIRECTORY / f"seed_{seed}"
    seed_checkpoint_directory.mkdir(parents=True, exist_ok=True)

    config.setdefault("checkpoint", {})
    config["checkpoint"]["directory"] = str(seed_checkpoint_directory.resolve())
    config["output_directory"] = str(OUTPUT_DIRECTORY)
    
    # Create temporary YAML path.
    temporary_path = TEMP_CONFIG_DIRECTORY/ f"seed_{seed}.yaml"

    save_yaml(config=config, path=temporary_path)

    # Verify that the generated configuration contains the expected path.
    saved_config = load_yaml(temporary_path)

    saved_checkpoint_directory = (
        saved_config
        .get("checkpoint", {})
        .get("directory")
    )

    expected_checkpoint_directory = str(
        seed_checkpoint_directory.resolve()
    )

    if saved_checkpoint_directory != expected_checkpoint_directory:
        raise RuntimeError(
            "Generated seed configuration has the wrong "
            "checkpoint directory.\n"
            f"Expected: {expected_checkpoint_directory}\n"
            f"Found:    {saved_checkpoint_directory}"
        )

    print(
        f"Seed {seed} checkpoint directory: "
        f"{expected_checkpoint_directory}"
    )

    print(
        f"Seed {seed} temporary config: "
        f"{temporary_path}"
    )

    return temporary_path


# Output parsing

def extract_metric(output, pattern):
    # Extract the final occurrence of a metric from training output

    matches = re.findall(pattern, output, flags=re.IGNORECASE)
    if not matches:
        return float("nan")

    return float(matches[-1])


def extract_best_checkpoint_path(output):
    # Extract the checkpoint path

    patterns = [
        r"Best checkpoint:\s*(.+?\.ckpt)",
        r"Restoring states from the checkpoint path at\s+(.+?\.ckpt)",
        r"Loaded model weights from the checkpoint at\s+(.+?\.ckpt)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            output,
            flags=re.IGNORECASE,
        )

        if matches:
            return matches[-1].strip()

    return ""


def extract_best_validation_f1(output):
    # Extracts the best validation F1.

    matches = re.findall(
        r"BEST_EPOCH\s+"
        r"best_epoch=(\d+)\s+"
        r"best_val_f1=([0-9.eE+-]+)",
        output,
        flags=re.IGNORECASE,
    )

    if not matches:
        return float("nan")

    _, best_f1 = matches[-1]

    return float(best_f1)


def extract_best_epoch(output):
    # Extract the epoch number.

    matches = re.findall(
            r"BEST_EPOCH\s+"
            r"best_epoch=(\d+)\s+"
            r"best_val_f1=([0-9.eE+-]+)",
            output,
            flags=re.IGNORECASE,
        )

    if not matches:
        return -1

    best_epoch, _ = matches[-1]

    return int(best_epoch)



def run_seed(seed, experiment_config_path, uncertainty_enabled):
    # Runs one complete training and test run on each seed

    command = [
        sys.executable,
        "-m",
        "train",
        "--base-config",
        str(BASE_CONFIG),
        "--experiment-config",
        str(experiment_config_path),
    ]

    print("\n" + "=" * 80)
    print(f"RUNNING SEED {seed}")
    print("=" * 80)
    print("Command:")
    print(" ".join(command))
    print()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines: list[str] = []

    if process.stdout is None:
        raise RuntimeError(
            "Unable to capture training output."
        )

    # Print output live and retain it for parsing.
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)

    return_code = process.wait()
    output = "".join(output_lines)

    log_path = LOG_DIRECTORY / f"seed_{seed}.log"

    log_path.write_text(output, encoding="utf-8")

    if return_code != 0:
        last_output = "".join(output_lines[-100:])

        raise RuntimeError(
            f"Seed {seed} failed with exit code "
            f"{return_code}.\n"
            f"Full log: {log_path}\n\n"
            f"Last 100 output lines:\n"
            f"{last_output}"
        )

    result = {
        "seed": int(seed),
        "best_epoch": extract_best_epoch(output),
        "best_val_f1": extract_best_validation_f1(output),
        "best_checkpoint": extract_best_checkpoint_path(output),
        "log_file": str(log_path),
    }
    

    for metric_name, pattern in METRIC_PATTERNS.items():
        result[metric_name] = extract_metric(output=output, pattern=pattern)

    required_metrics = BASE_REQUIRED_METRICS + CALIBRATION_METRICS

    if uncertainty_enabled:
        required_metrics.extend(UNCERTAINTY_METRICS)
        
    missing_metrics = [
        metric
        for metric in required_metrics
        if math.isnan(float(result[metric]))
    ]

    if missing_metrics:
        raise RuntimeError(
            f"Seed {seed} finished, but these test metrics "
            f"could not be extracted: {missing_metrics}.\n"
            f"Inspect: {log_path}"
        )

    print("\n" + "-" * 80)
    print(f"SEED {seed} COMPLETED")
    print("-" * 80)
    print(f"Best epoch:        {result['best_epoch']}")
    print(f"Best val F1:       {float(result['best_val_f1']):.4f}")
    print(f"Test F1:           {float(result['test_f1']):.4f}")
    print(f"Test accuracy:     {float(result['test_accuracy']):.4f}")
    print(f"Test precision:    {float(result['test_precision']):.4f}")
    print(f"Test recall:       {float(result['test_recall']):.4f}")
    print(f"Test AUROC:        {float(result['test_auroc']):.4f}")
    print(f"Test Brier:        {float(result['test_brier']):.4f}")
    print(f"Test ECE:          {float(result['test_ece']):.4f}")

    return result



def save_results(results):
    # Save all completed seed runs to a CSV file

    if not results:
        return

    fieldnames = [
        "seed",
        "best_epoch",
        "best_val_f1",

        # Classification
        "test_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_auroc",
        "test_loss",

        # Calibration
        "test_ece",
        "test_brier",

        # EDL uncertainty
        "test_mean_uncertainty",
        "test_aurc"

        "best_checkpoint",
        "log_file",
    ]

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)


def calculate_summary(results):
    """
    Calculate sample statistics across completed runs.
    Standard deviation and variance use n - 1.
    """

    metrics = [
        "best_val_f1",
        "test_f1",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_auroc",
        "test_loss",
        "test_mean_uncertainty",
        "test_ece",
        "test_brier",
        "test_aurc",
    ]

    summary = []

    for metric in metrics:
        valid_results = []

        for result in results:
            value = result.get(metric)

            if not isinstance(value, (int, float)):
                continue

            numeric_value = float(value)

            if math.isnan(numeric_value):
                continue

            valid_results.append(result)

        if not valid_results:
            continue

        values = [float(result[metric]) for result in valid_results]

        mean = statistics.mean(values)

        standard_deviation = (
            statistics.stdev(values)
            if len(values) > 1
            else 0.0
        )

        variance = (
            statistics.variance(values)
            if len(values) > 1
            else 0.0
        )

        minimum_result = min(
            valid_results,
            key=lambda result: float(result[metric]),
        )

        maximum_result = max(
            valid_results,
            key=lambda result: float(result[metric]),
        )

        minimum = float(minimum_result[metric])
        maximum = float(maximum_result[metric])

        summary.append(
            {
                "metric": metric,
                "runs": len(values),
                "mean": mean,
                "standard_deviation": standard_deviation,
                "variance": variance,
                "minimum": minimum,
                "minimum_seed": int(
                    minimum_result["seed"]
                ),
                "maximum": maximum,
                "maximum_seed": int(
                    maximum_result["seed"]
                ),
                "mean_plus_minus_std": (
                    f"{mean:.4f} "
                    f"± {standard_deviation:.4f}"
                ),
            }
        )

    return summary



def save_summary(summary):
    # Saves aggregate statistics to a CSV file

    if not summary:
        return

    fieldnames = [
        "metric",
        "runs",
        "mean",
        "standard_deviation",
        "variance",
        "minimum",
        "minimum_seed",
        "maximum",
        "maximum_seed",
        "mean_plus_minus_std",
    ]

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerows(summary)



def print_summary(results, summary, uncertainty_enabled=False):

    # Print individual seed results and aggregate results.

    print("\n" + "=" * 100)
    print("MULTI-SEED RESULTS")
    print("=" * 100)

    print(
        f"{'Seed':>7}"
        f"{'Epoch':>9}"
        f"{'Val F1':>12}"
        f"{'Test F1':>12}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'AUROC':>12}"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{int(result['seed']):>7}"
            f"{int(result['best_epoch']):>9}"
            f"{float(result['best_val_f1']):>12.4f}"
            f"{float(result['test_f1']):>12.4f}"
            f"{float(result['test_accuracy']):>12.4f}"
            f"{float(result['test_precision']):>12.4f}"
            f"{float(result['test_recall']):>12.4f}"
            f"{float(result['test_auroc']):>12.4f}"
        )

    print("\nAggregate results")
    print("-" * 100)

    for row in summary:
        print(
            f"{str(row['metric']):<18}: "
            f"{row['mean_plus_minus_std']} "
            f"| variance="
            f"{float(row['variance']):.8f} "
            f"| min="
            f"{float(row['minimum']):.4f} "
            f"(seed={int(row['minimum_seed'])}) "
            f"| max="
            f"{float(row['maximum']):.4f} "
            f"(seed={int(row['maximum_seed'])}) "
            f"| runs={int(row['runs'])}"
        )

    test_f1_summary = next(
        row
        for row in summary
        if row["metric"] == "test_f1"
    )

    validation_f1_summary = next(
        row
        for row in summary
        if row["metric"] == "best_val_f1"
    )

    print("\n" + "=" * 100)
    print("CALIBRATION RESULTS")
    print("=" * 100)

    for metric in CALIBRATION_METRICS:
        row = next(
            (item for item in summary if item["metric"] == metric),
            None,
        )

        if row is not None:
            print(f"{metric:<30}: {row['mean_plus_minus_std']}")

    if uncertainty_enabled:
        print("\n" + "=" * 100)
        print("UNCERTAINTY RESULTS")
        print("=" * 100)
  
        for metric in UNCERTAINTY_METRICS:
            row = next((item for item in summary if item["metric"] == metric), None)
            if row is not None:
                print(f"{metric:<30}: {row['mean_plus_minus_std']}")


    print("\n" + "=" * 100)
    print("FINAL REPORTED RESULTS")
    print("=" * 100)

    print(
        "Validation F1 = "
        f"{validation_f1_summary['mean_plus_minus_std']} "
        f"(runs={validation_f1_summary['runs']})"
    )

    print(
        "Test F1       = "
        f"{test_f1_summary['mean_plus_minus_std']} "
        f"(runs={test_f1_summary['runs']})"
    )

    print(
        f"Test F1 range = {float(test_f1_summary['minimum']):.4f} "
        f"to {float(test_f1_summary['maximum']):.4f}"
    )

    print(f"\nPer-seed results: {RESULTS_CSV}")
    print(f"Aggregate summary: {SUMMARY_CSV}")



def main() -> None:
    # Run all configured seeds sequentially.

    global BASE_CONFIG
    global EXPERIMENT_CONFIG
    global OUTPUT_DIRECTORY
    global TEMP_CONFIG_DIRECTORY
    global LOG_DIRECTORY
    global CHECKPOINT_DIRECTORY
    global RESULTS_CSV
    global SUMMARY_CSV

    parser = argparse.ArgumentParser(description="Run a multi-seed training experiment.")

    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="Path to the base configuration YAML.",
    )

    parser.add_argument(
        "--experiment-config",
        type=Path,
        required=True,
        help="Path to the experiment configuration YAML.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory used to store experiment outputs.",
    )

    args = parser.parse_args()

    # Configure experiment paths
    BASE_CONFIG = args.base_config
    EXPERIMENT_CONFIG = args.experiment_config
    OUTPUT_DIRECTORY = args.output_dir

    TEMP_CONFIG_DIRECTORY = OUTPUT_DIRECTORY / "temporary_configs"
    LOG_DIRECTORY = OUTPUT_DIRECTORY / "logs"
    CHECKPOINT_DIRECTORY = OUTPUT_DIRECTORY / "checkpoints"

    RESULTS_CSV = OUTPUT_DIRECTORY / "results.csv"
    SUMMARY_CSV = OUTPUT_DIRECTORY / "summary.csv"

    # Create output directories
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    TEMP_CONFIG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Check configuration files
    if not BASE_CONFIG.exists():
        raise FileNotFoundError(
            f"Base configuration not found: "
            f"{BASE_CONFIG.resolve()}"
        )

    if not EXPERIMENT_CONFIG.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: "
            f"{EXPERIMENT_CONFIG.resolve()}"
        )

    print("\n" + "=" * 100)
    print("MULTI-SEED EXPERIMENT")
    print("=" * 100)
    print(f"Base config       : {BASE_CONFIG}")
    print(f"Experiment config : {EXPERIMENT_CONFIG}")
    print(f"Output directory  : {OUTPUT_DIRECTORY}")
    print(f"Seeds             : {len(SEEDS)}")
    print("=" * 100)

    original_config = load_yaml(EXPERIMENT_CONFIG)

    uncertainty_enabled = (
        original_config
        .get("uncertainty_opts", {})
        .get("enabled", False)
    )

    print(f"Uncertainty : {'enabled' if uncertainty_enabled else 'disabled'}")

    results = []

    for run_number, seed in enumerate(SEEDS, start=1):
        print("\n" + "*" * 100)
        print(f"STARTING RUN {run_number}/{len(SEEDS)} — SEED {seed}")
        print("*" * 100)

        temporary_config = create_seed_config(original_config=original_config, seed=seed)

        try:
            result = run_seed(
                seed=seed,
                experiment_config_path=temporary_config,
                uncertainty_enabled=uncertainty_enabled,
            )

            results.append(result)

            # Preserve completed runs immediately
            save_results(results)
            partial_summary = calculate_summary(results)
            save_summary(partial_summary)

        except Exception as error:
            print("\n" + "!" * 100)
            print(f"SEED {seed} FAILED")
            print("!" * 100)
            print(error)

            if results:
                save_results(results)

                partial_summary = calculate_summary(results)
                save_summary(partial_summary)

                print_summary(results=results, summary=partial_summary, uncertainty_enabled=uncertainty_enabled)

            raise

    summary = calculate_summary(results)

    save_results(results)
    save_summary(summary)

    print_summary(results=results, summary=summary, uncertainty_enabled=uncertainty_enabled)

    # Remove temporary config directory after all seeds finish
    if TEMP_CONFIG_DIRECTORY.exists():
        shutil.rmtree(TEMP_CONFIG_DIRECTORY)

if __name__ == "__main__":
    main()