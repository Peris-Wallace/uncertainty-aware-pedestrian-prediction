# Import libraries
from pathlib import Path
import numpy as np


from evaluation.plots import (
    plot_aggregate_predictive_performance,
    plot_aggregate_reliability,
    plot_reliability_bars,
    plot_aggregate_uncertainty_performance,
    plot_aggregate_risk_coverage,
    plot_aggregate_precision_recall,
    plot_aggregate_confusion_matrices,
)


def load_outputs(folder):
    # Load seed test outputs
    files = sorted(
        Path(folder).glob("test_outputs/seed_*.npz"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    print(f"Found {len(files)} runs in {folder}")
    return [np.load(file) for file in files]

def combine_outputs(runs):
    # Pool predictions and targets from all seeds
    # Used for the reliability diagram
    probabilities = np.concatenate([run["probabilities"] for run in runs], axis=0)
    targets = np.concatenate([run["targets"].reshape(-1) for run in runs], axis=0)
    return probabilities, targets


# Load models
det_model = load_outputs("results/flip_only/obs_15_8_area_det")
edl_model = load_outputs("results/flip_only/obs_15_8_area_edl_mse")


# Verify complete multi-seed evaluation
assert len(det_model) == 16
assert len(edl_model) == 16


output_dir = Path("results/aggregate_plots")
output_dir.mkdir(parents=True, exist_ok=True)


plots = [
    (plot_aggregate_predictive_performance, (det_model, edl_model), "01_predictive_performance.png"),
    (plot_aggregate_reliability, (det_model, edl_model), "02_calibration_line_plot.png"),
    (plot_aggregate_uncertainty_performance, (edl_model,), "03_uncertainty_performance.png"),
    (plot_aggregate_risk_coverage, (edl_model,), "04_risk_coverage.png"),
    (plot_aggregate_precision_recall, (det_model, edl_model), "05_precision_recall.png"),
    (plot_aggregate_confusion_matrices, (det_model, edl_model), "06_confusion_matrices.png"),
]

for function, args, filename in plots:
    function(*args, save_path=output_dir / filename)


# Reliability diagrams
det_probs, det_targets = combine_outputs(det_model)
edl_probs, edl_targets = combine_outputs(edl_model)

plot_reliability_bars(
    probs=det_probs,
    targets=det_targets,
    save_path=output_dir / "07_det_reliability_bar.png",
    n_bins=10,
    title="Deterministic",
)


plot_reliability_bars(
    probs=edl_probs,
    targets=edl_targets,
    save_path=output_dir / "08_edl_reliability_bar.png",
    n_bins=10,
    title="Evidential",
)


print(f"Saved aggregate figures to {output_dir.resolve()}")