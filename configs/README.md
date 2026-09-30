## Configuration Guide

Experiments are configured by combining:

configs/base.yaml
+
configs/experiments/<experiment>.yaml

The base file contains settings shared across experiments. The experiment file contains only
the values that change for a particular condition. When the files are merged, values from the
experiment configuration override matching values in base.yaml.

### 1. Set the PIE dataset path

The default example assumes:

pie_data:
  raw_data_directory: data/PIE

If PIE is stored elsewhere, update this path in configs/base.yaml.

Cache, checkpoint, and MLflow paths in the example configuration are relative to the repository
root so that no user-specific /home/... paths are required.

### 2. Run an example

Deterministic:

python train.py \
    --base-config configs/base.yaml \
    --experiment-config configs/experiments/example_deterministic.yaml

Evidential:

python train.py \
    --base-config configs/base.yaml \
    --experiment-config configs/experiments/example_evidential.yaml

### Observation length

The dissertation reports final observation sequences of 15 or 30 frames. The preprocessing
pipeline requires one additional raw frame before relative coordinate normalisation.

Therefore:

Final sequence length 15 -> obs_length: 16
Final sequence length 30 -> obs_length: 31

The first relative observation is zero by construction and is removed after normalisation.

The corresponding minimum track sizes are:

obs_length 16 + max(TTE) 60 = 76
obs_length 31 + max(TTE) 60 = 91

### Main experiment controls

**Feature configuration**

feature_opts:
  use_tabular: true
  use_center: true
  use_area: true
  use_aspect_ratio: false

With bounding-box coordinates and ego speed included, this produces the eight-feature
representation used in the principal experiments.

**Crossing-class augmentation**

Enable training-only horizontal mirroring:

model_opts:
  balance_data: true
  balance_strategy: flip_only

Disable augmentation:

model_opts:
  balance_data: false

Validation and test data are not augmented.

**Deterministic model**

uncertainty_opts:
  enabled: false

**Evidential model**

uncertainty_opts:
  enabled: true
  loss: mse
  annealing_step: 10

**Temporal** 

model_opts:
  time_to_event: [30, 60]
  overlap: 0.5

At 30 FPS, the TTE interval places the end of the observation approximately 1--2 seconds before
the crossing event.

**Cache controls**

The pipeline supports both behaviour-sequence caches and processed-feature caches:

pie_data:
  cache_enabled: true
  use_cached_sequences: true
  regenerate_sequences: false
  use_cached_processed: false
  regenerate_processed: true
  strict_cache: true

For a first run, you may need to regenerate the required caches. Once they exist, cached data can
be reused by changing the relevant use_cached_* and regenerate_* flags.

Keep separate processed cache directories for experimental conditions that alter preprocessing,
observation length, or augmentation.

**Checkpoints**

Example experiment files define their own checkpoint directories:

checkpoint:
  directory: checkpoints/example_deterministic

This prevents deterministic and evidential runs from overwriting one another.

**MLflow**

The default local tracking database is:

mlflow:
  tracking_uri: sqlite:///mlflow.db

This creates mlflow.db in the repository root.

**Multi-seed experiments**

For the dissertation experiments, the following seeds were used:

42, 101, 202, 303, 404, 505, 606, 707,
808, 909, 1010, 1111, 1212, 1313, 1414, 1515

Use scripts/run_multi_seed.py to execute the same experiment across the configured seed list.
