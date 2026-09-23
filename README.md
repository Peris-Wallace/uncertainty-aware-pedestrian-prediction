# Uncertainty-Aware Pedestrian Crossing Prediction

This repository contains the implementation for an uncertainty-aware pedestrian
crossing prediction framework using Transformer-based temporal modelling and
Evidential Deep Learning (EDL).

The framework is evaluated on the **Pedestrian Intention Estimation (PIE)**
dataset using structured pedestrian geometry and ego-vehicle information.

## Problem Statement

Pedestrian crossing prediction is an important task in autonomous driving,
where a vehicle must anticipate whether a pedestrian is likely to enter the
road before the crossing action becomes directly observable.

Predictive performance alone does not indicate whether a model's confidence is
reliable. This project therefore investigates both pedestrian crossing
classification and predictive uncertainty.

The framework compares a conventional deterministic Transformer with an
equivalent Evidential Deep Learning model and evaluates predictive performance,
probability calibration and uncertainty-based reliability.

## Method Overview

Each pedestrian is represented as a temporal sequence containing combinations
of:

- bounding-box coordinates;
- pedestrian centre coordinates;
- bounding-box area;
- bounding-box aspect ratio; and
- ego-vehicle speed.

A lightweight Transformer encoder models the temporal sequence.

Two prediction formulations are supported:

- **Deterministic:** cross-entropy classification with softmax probabilities.
- **Evidential:** Dirichlet-based Evidential Deep Learning providing both class
  probabilities and predictive uncertainty.

Experiments also investigate feature representation, observation length,
training augmentation and variation across random seeds.

## Repository Structure

```text
.
├── configs/                    # Base and experiment configurations
├── PedestrianActionBenchmark/ # PIE data-loading utilities
├── data/                       # Cached PIE sequences and processed data
├── evaluation/                 # Metrics and evaluation plots
├── results/                    # Experimental results
├── scripts/                    # Multi-seed experiment utilities
├── training/                   # Dataset, model and training components
├── train.py                    # Main training entry point
├── utils.py                    # General utilities
└── requirements.txt            # Python dependencies
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd uncertainty-aware-pedestrian-prediction
```

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The PIE dataset should be obtained separately and its location specified in the
experiment configuration.

## Configuration

Experiments use two YAML configuration files:

```text
configs/base.yaml
+
configs/experiments/<experiment>.yaml
```

`base.yaml` contains settings shared across experiments, while the experiment
configuration specifies the condition being evaluated.

Experiment-specific values override the corresponding values in the base
configuration.

For further information on configuring experiments, see:

[`configs/README.md`](configs/README.md)

## Running a Single Experiment

Run experiments from the repository root:

```bash
python train.py \
    --base-config configs/base.yaml \
    --experiment-config configs/experiments/example.yaml
```

The training pipeline automatically:

1. loads and merges the configuration files;
2. configures reproducibility;
3. prepares and verifies the PIE dataset;
4. constructs the Transformer model;
5. trains and validates the model;
6. selects the best validation checkpoint; and
7. evaluates the selected model on the held-out test set.

## Running Multi-Seed Experiments

Repeated experiments can be executed using:

```bash
python scripts/run_multi_seed.py \
    --base-config configs/base.yaml \
    --experiment-config configs/experiments/example.yaml \
    --output-dir results/example
```

The multi-seed runner creates an independent configuration and checkpoint
directory for each seed and produces both per-seed and aggregate results.

Typical outputs include:

```text
results/example/
├── checkpoints/
├── logs/
├── results.csv
└── summary.csv
```

## Evaluation

The framework reports predictive metrics including:

- F1 score;
- accuracy;
- precision;
- recall; and
- AUROC.

Probability reliability is evaluated using:

- Expected Calibration Error (ECE); and
- Brier score.

For evidential models, uncertainty behaviour is additionally evaluated using
uncertainty statistics and risk-coverage analysis.

## Reproducibility

The experimental pipeline supports reproducibility through:

- YAML-based experiment configurations;
- deterministic random seeding;
- fixed training, validation and test splits;
- saved resolved configurations;
- independent checkpoints for each random seed;
- MLflow experiment tracking; and
- repeated multi-seed evaluation.

The validation set is used for model selection and decision-threshold tuning.
The held-out test set is used only for final evaluation.

## Acknowledgements

The PIE data preparation pipeline is based in part on the
[PedestrianActionBenchmark](https://github.com/aras62/PIEPredict) utilities.

Parts of the preprocessing implementation were adapted from the
[TrEP](https://github.com/zzmonlyyou/TrEP) implementation:

> Zhang, Z., Tian, R. and Ding, Z. (2023).  
> *TrEP: Transformer-Based Evidential Prediction for Pedestrian Intention with
> Uncertainty.*  
> Proceedings of the AAAI Conference on Artificial Intelligence.
