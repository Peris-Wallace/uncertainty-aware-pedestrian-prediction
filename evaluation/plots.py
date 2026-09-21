from pathlib import Path

# Import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from torchmetrics.functional.classification import (
    binary_f1_score,
    binary_precision,
    binary_recall,
)


COLOURS = {
    "navy": "#163A5F",
    "blue": "#2563A6",
    "medium_blue": "#4C8CCB",
    "light_blue": "#9CC5E8",
    "pale_blue": "#EAF3FA",
    "dark": "#20252B",
    "grey": "#667085",
    "light_grey": "#D9E1E8",
    "white": "#FFFFFF",
    "det_colour": "#6B7280",
    "edl_colour": "#2A9D8F", 
}


def set_plot_style():
    # Define a consistent plotting theme
    sns.set_theme(style="whitegrid", context="paper")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],

        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,

        "figure.figsize": (5.2, 3.8),
        "figure.dpi": 120,
        "savefig.dpi": 300,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLOURS["light_grey"],
        "axes.linewidth": 0.8,

        "axes.labelcolor": COLOURS["dark"],
        "axes.titlecolor": COLOURS["dark"],
        "xtick.color": COLOURS["grey"],
        "ytick.color": COLOURS["grey"],

        "axes.grid": True,
        "grid.color": COLOURS["light_grey"],
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,

        "axes.facecolor": COLOURS["white"],
        "figure.facecolor": COLOURS["white"],
        "lines.linewidth": 1.4,
    })

set_plot_style()


def plot_validation_f1(
    epochs,
    values,
    best_epoch=None,
    title="Validation F1 Across Training",
):
    # Plot validation F1 across training epochs.

    epochs = np.asarray(epochs)
    values = np.asarray(values)

    if epochs.ndim != 1 or values.ndim != 1:
        raise ValueError("epochs and values must both be one-dimensional.")

    if epochs.size != values.size:
        raise ValueError("epochs and values must contain the same number of elements.")

    fig, ax = plt.subplots()

    ax.plot(
        epochs,
        values,
        color=COLOURS["blue"],
        label="Validation F1",
    )

    if best_epoch is None:
        best_index = int(np.argmax(values))
        selected_best_epoch = epochs[best_index]
    else:
        best_index = int(
            np.argmin(
                np.abs(
                    epochs - best_epoch
                )
            )
        )
        selected_best_epoch = epochs[best_index]

    best_f1 = values[best_index]

    ax.scatter(
        selected_best_epoch,
        best_f1,
        color=COLOURS["navy"],
        s=55,
        zorder=3,
    )

    ax.axvline(
        selected_best_epoch,
        color=COLOURS["medium_blue"],
        linestyle="--",
        linewidth=1.3,
        alpha=0.8,
    )

    ax.annotate(
        (
            f"Best F1 = {best_f1:.3f}\n"
            f"Epoch {selected_best_epoch}"
        ),
        xy=(selected_best_epoch, best_f1),
        xytext=(12, -35),
        textcoords="offset points",
        fontsize=9,
        color=COLOURS["navy"],
        arrowprops={
            "arrowstyle": "-",
            "color": COLOURS["medium_blue"],
        },
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")

    if title:
        ax.set_title(title)

    ax.legend(frameon=False)

    fig.tight_layout()

    return fig, ax


def plot_validation_model_selection(
    epochs,
    f1_values,
    thresholds,
    best_epoch,
    best_f1,
    best_threshold,
):
    epochs = np.asarray(epochs)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(4.8, 3.6),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )

    # Validation F1
    ax1.plot(epochs, f1_values, linewidth=1.4)
    ax1.axvline(best_epoch, linestyle="--", linewidth=0.9, alpha=0.65)
    ax1.scatter(best_epoch, best_f1, s=32, zorder=5)
    # Set title properties 
    ax1.set_title(
        "Validation F1 Across Training Epochs",
        pad=24,
    )
    ax1.set_ylabel("F1") 
    ax1.annotate(
        f"Epoch {best_epoch}\nF1 = {best_f1:.3f}",
        (best_epoch, best_f1),
        xytext=(12, 8),
        textcoords="offset points",
    )

    # Decision threshold
    ax2.plot(epochs, thresholds, linewidth=1.0, alpha=0.85)
    ax2.axvline(best_epoch, linestyle="--", linewidth=0.9, alpha=0.65)
    ax2.axhline(0.50, linestyle=":", linewidth=0.8, alpha=0.55)
    ax2.scatter(best_epoch, best_threshold, s=32, zorder=5)

    # Set title properties directly using set_title
    ax2.set_title(
        "Selected Decision Threshold",
        pad=14,
    )
    ax2.set_ylabel("Threshold", fontsize=8.5)
    ax2.set_xlabel("Epoch", fontsize=8.5)

    ax2.annotate(
        rf"$\tau^* = {best_threshold:.2f}$",
        (best_epoch, best_threshold),
        xytext=(8, 30),
        textcoords="offset points",
        fontsize=8,
    )

    fig.subplots_adjust(
        top=0.95,
        bottom=0.11,
        left=0.12,
        right=0.98,
    )

    return fig


def plot_confusion_matrix(
    confusion_matrix,
    class_labels=("Non-crossing", "Crossing"),
    normalize=True,
    title="Test Confusion Matrix",
):
    """Plot a binary confusion matrix.
    If normalize=True, row-normalize the confusion matrix."""
    cm = np.asarray(confusion_matrix)

    if cm.shape != (2, 2):
        raise ValueError("Expected a 2x2 confusion matrix.")

    values = (
        cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        if normalize else cm.astype(float)
    )

    annotations = np.array([
        [
            f"{values[r, c]:.1%}\n(n={cm[r, c]})"
            if normalize else str(cm[r, c])
            for c in range(2)
        ]
        for r in range(2)
    ])

    fig, ax = plt.subplots(figsize=(3.6, 3.1))

    sns.heatmap(
        values,
        annot=annotations,
        fmt="",
        cmap="Blues",
        vmin=0,
        vmax=1 if normalize else None,
        square=True,
        linewidths=0,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=False,
        ax=ax,
    )

    ax.set(
        title=title,
        xlabel="Predicted class",
        ylabel="Actual class",
    )
    ax.tick_params(rotation=0)

    fig.tight_layout()
    return fig, ax


def plot_precision_recall_curve(
    probabilities,
    targets,
    selected_threshold=None,
    selected_precision=None,
    selected_recall=None,
    title="Test Precision–Recall Curve",
):
    # Plot the precision-recall trade-off across all decision thresholds.

    probabilities = np.asarray(probabilities).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    if probabilities.size != targets.size:
        raise ValueError("probabilities and targets must contain be of the same length")


    precision, recall, _ = precision_recall_curve(targets, probabilities)
    
    # Average precision score
    ap = average_precision_score(targets, probabilities)

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    ax.plot(
        recall,
        precision,
        color=COLOURS["blue"],
    )

    # Mark the operating point selected using validation data.
    if selected_precision is not None and selected_recall is not None:
        ax.scatter(
            float(selected_recall),
            float(selected_precision),
            s=35,
            color=COLOURS["navy"],
            zorder=4,
        )

        annotation = (
            f"Precision = {float(selected_precision):.3f}\n"
            f"Recall = {float(selected_recall):.3f}"
        )

        if selected_threshold is not None:
            annotation += f"\n$\\tau$ = {selected_threshold:.2f}"

        ax.annotate(
            annotation,
            xy=(
                float(selected_recall),
                float(selected_precision),
            ),
            xytext=(-70, -35),
            textcoords="offset points",
            fontsize=7.5,
            arrowprops={
                "arrowstyle": "-",
                "color": COLOURS["medium_blue"],
            },
        )

    ax.text(
        0.04,
        0.06,
        f"AP = {ap:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
    )

    ax.set(
        title=title,
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )


    fig.tight_layout()
    return fig, ax


def plot_reliability_diagram(
    probabilities,
    targets,
    n_bins=10,
    ece=None,
    title="Test Calibration",
):
    # Plots confidence against accuracy
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets).reshape(-1)

    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == targets).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    accuracy = np.full(n_bins, np.nan)
    mean_confidence = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (
            (confidence >= edges[i]) & (confidence <= edges[i + 1])
            if i == 0
            else (confidence > edges[i]) & (confidence <= edges[i + 1])
        )

        if mask.any():
            accuracy[i] = correct[mask].mean()
            mean_confidence[i] = confidence[mask].mean()

    valid = ~np.isnan(accuracy)

    fig, ax = plt.subplots(figsize=(3.9, 3.2))

    ax.plot(
        [0, 1], [0, 1],
        "--",
        color=COLOURS["grey"],
        linewidth=1,
    )

    ax.bar(
        centers[valid],
        accuracy[valid],
        width=0.085,
        color=COLOURS["light_blue"],
        edgecolor=COLOURS["blue"],
        alpha=0.8,
    )

    ax.plot(
        mean_confidence[valid],
        accuracy[valid],
        "o-",
        color=COLOURS["navy"],
        markersize=4,
        linewidth=1.2,
    )

    if ece is not None:
        ax.text(
            0.04,
            0.95,
            f"ECE = {ece:.3f}",
            transform=ax.transAxes,
            va="top",
        )

    ax.set(
        title=title,
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()
    return fig, ax


def plot_risk_coverage(
    coverage,
    risk,
    aurc=None,
    title="Test Risk–Coverage",
):
    """
    Plot selective prediction risk against coverage.

    Lower curves indicate that uncertain predictions
    can be rejected while retaining low-risk predictions.
    """
    coverage = np.asarray(coverage).reshape(-1)
    risk = np.asarray(risk).reshape(-1)

    fig, ax = plt.subplots(figsize=(4.2, 3.1))

    ax.plot(
        coverage,
        risk,
        color=COLOURS["blue"],
    )

    ax.fill_between(
        coverage,
        risk,
        color=COLOURS["medium_blue"],
        alpha=0.10,
    )

    if aurc is not None:
        ax.text(
            0.96,
            0.94,
            f"AURC = {aurc:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    ax.set(
        title=title,
        xlabel="Coverage",
        ylabel="Risk (1 − accuracy)",
        xlim=(0, 1),
    )
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax



def plot_uncertainty_distribution(
    uncertainties,
    bins=25,
    kde=True,
    title="Test Evidential Uncertainty",
):
    # Plot the distribution of evidential uncertainty.
    uncertainties = np.asarray(
        uncertainties,
        dtype=float,
    ).reshape(-1)

    mean_u = uncertainties.mean()

    fig, ax = plt.subplots(figsize=(4.2, 3.1))

    sns.histplot(
        uncertainties,
        bins=bins,
        kde=kde,
        color=COLOURS["medium_blue"],
        edgecolor=COLOURS["white"],
        alpha=0.75,
        ax=ax,
    )

    ax.axvline(
        mean_u,
        color=COLOURS["navy"],
        linestyle="--",
        linewidth=1.2,
        label=f"Mean = {mean_u:.3f}",
    )

    ax.set(
        title=title,
        xlabel="Evidential uncertainty",
        ylabel="No. of Samples",
    )

    ax.legend(frameon=False)
    fig.tight_layout()

    return fig, ax


def plot_metric_history(
    epochs,
    values,
    ylabel,
    title=None,
    best_epoch=None,
):
     # Plots validation/training metrics against epoch.
    fig, ax = plt.subplots(figsize=(4.4, 2.8))

    ax.plot(
        epochs,
        values,
        color=COLOURS["blue"],
    )

    if best_epoch is not None:
        ax.axvline(
            best_epoch,
            linestyle="--",
            linewidth=0.9,
            color=COLOURS["medium_blue"],
            alpha=0.7,
        )

    ax.set(
        title=title,
        xlabel="Epoch",
        ylabel=ylabel,
    )

    fig.tight_layout()
    return fig, ax


def save_figure(fig, path_without_extension):

    # Save plot locally
    path_without_extension = Path(path_without_extension)
    path_without_extension.parent.mkdir(parents=True, exist_ok=True)

    png_path = path_without_extension.with_suffix(".png")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    return png_path


def log_figure(module, fig, filename, artifact_path="plots"):
    # Saves a figure as PNG logs to the current MLflow run.

    if module.logger is None:
        plt.close(fig)
        return

    output_directory = Path(module.trainer.default_root_dir) / "plot_artifacts"
    output_directory.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    png_path = output_directory / f"{stem}.png"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    experiment = module.logger.experiment
    run_id = module.logger.run_id
    
    experiment.log_artifact(
        run_id,
        str(png_path),
        artifact_path=artifact_path,
    )

    plt.close(fig)


######################
# Aggregate figures
######################


def _mean_std(runs, key):
    # Mean and SD of one scalar metric across seeds
    values = np.array([run[key].item() for run in runs], dtype=float)
    return values.mean(), values.std(ddof=1)


def plot_aggregate_predictive_performance(det_runs, edl_runs, save_path):
    # Compare predictive metrics across seeds

    metrics = ["f1", "accuracy", "precision", "recall", "auroc"]
    labels = ["F1", "Accuracy", "Precision", "Recall", "AUROC"]

    det = np.array([_mean_std(det_runs, m) for m in metrics])
    edl = np.array([_mean_std(edl_runs, m) for m in metrics])

    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(5.4, 3.4))

    ax.errorbar(
        x - 0.07,
        det[:, 0],
        yerr=det[:, 1],
        fmt="o",
        markersize=7,
        capsize=4,
        elinewidth=1.6,
        color= COLOURS["det_colour"],
        label="Deterministic",
    )

    ax.errorbar(
        x + 0.07,
        edl[:, 0],
        yerr=edl[:, 1],
        fmt="o",
        markersize=7,
        capsize=4,
        elinewidth=1.6,
        color= COLOURS["edl_colour"],
        label="Evidential",
    )

    ax.set_xticks(x, labels)
    ax.set(ylabel="Score",ylim=(0.75, 0.96))
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_bars(probs, targets, save_path, n_bins=10, title=None):
    # Reliability diagram with bars + calibration-gap shading

    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == targets.reshape(-1)).astype(float)

    edges = np.linspace(0, 1, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    mean_conf = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)

    # Compute mean confidence and observed accuracy in each bin
    for i in range(n_bins):
        if i == 0:
            mask = (confidence >= edges[i]) & (confidence <= edges[i + 1])
        else:
            mask = (confidence > edges[i]) & (confidence <= edges[i + 1])

        if mask.any():
            mean_conf[i] = confidence[mask].mean()
            accuracy[i] = correct[mask].mean()

    valid = ~np.isnan(mean_conf) & ~np.isnan(accuracy)
    x = centres[valid]
    conf = mean_conf[valid]
    acc = accuracy[valid]

    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    # Perfect calibration
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.5)

    # Observed accuracy bars
    ax.bar(
        x,
        acc,
        width=width,
        align="center",
        edgecolor="black",
        linewidth=0.8,
        label="Outputs",
    )

    # Calibration gap shading
    gap_bottom = np.minimum(acc, conf)
    gap_height = np.abs(conf - acc)

    ax.bar(
        x,
        gap_height,
        width=width,
        bottom=gap_bottom,
        align="center",
        hatch="/",
        alpha=0.35,
        edgecolor="red",
        linewidth=0.8,
        label="Calibration gap",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")

    if title is not None:
        ax.set_title(title)

    ax.legend(loc="upper left")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_reliability(det_runs, edl_runs, save_path, n_bins=15):
    # Function for plotting mean reliability curves with +/- 1 SD
    
    # Define confidence-bin boundaries from 0 to 1
    edges = np.linspace(0, 1, n_bins + 1)

    def curves(runs):
        # Store one reliability curve per seed
        confs, accs = [], []

        for run in runs:
            probs = run["probabilities"]
            targets = run["targets"].reshape(-1)

            # confidence is the probability assigned to the predicited class
            confidence = probs.max(axis=1)
            
            # if correct prediction
            correct = (probs.argmax(axis=1) == targets).astype(float)

            # Use NaN for bins containing no samples
            mean_conf = np.full(n_bins, np.nan)
            accuracy = np.full(n_bins, np.nan)
            
            # Compute mean confidence and accuracy in each bin
            for i in range(n_bins):
                if i == 0:
                    mask = (
                        (confidence >= edges[i])
                        & (confidence <= edges[i + 1])
                    )
                else:
                    mask = (
                        (confidence > edges[i])
                        & (confidence <= edges[i + 1])
                    )

                if mask.any():
                    mean_conf[i] = confidence[mask].mean()
                    accuracy[i] = correct[mask].mean()

            confs.append(mean_conf)
            accs.append(accuracy)

        confs = np.stack(confs)
        accs = np.stack(accs)

        # Return mean curve and between-seed variability
        return (
            np.nanmean(confs, axis=0),
            np.nanmean(accs, axis=0),
            np.nanstd(accs, axis=0, ddof=1),
        )

    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    # Diagonal reference for perfect calibration
    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color=COLOURS["light_grey"],
        label="Perfect calibration",
    )

    for runs, label, colour in [
        (det_runs, "Deterministic", COLOURS["det_colour"]),
        (edl_runs, "Evidential", COLOURS["edl_colour"]),
    ]:
        x, y, sd = curves(runs)
                
        # Ignore confidence bins that were empty across seeds
        valid = ~np.isnan(x) & ~np.isnan(y)

        # Mean reliability curve across 16 seeds
        ax.plot(
            x[valid],
            y[valid],
            "o-",
            markersize=3,
            color=colour,
            label=label,
        )

        # Shaded region shows +/- 1 standard deviation across seeds
        ax.fill_between(
            x[valid],
            np.clip(y[valid] - sd[valid], 0, 1),
            np.clip(y[valid] + sd[valid], 0, 1),
            color=colour,
            alpha=0.15,
        )

    ax.set(
        xlabel="Mean confidence",
        ylabel="Observed accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.legend(frameon=False)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _binary_metrics(probabilities, targets, mask, threshold):
    # Compute Precision, recall, F1 and accuracy for selected samples

    # Return NaN if the uncertainty bin contains no samples
    if not mask.any():
        return np.full(4, np.nan)

    y_prob = torch.as_tensor(probabilities[mask], dtype=torch.float32)
    y_true = torch.as_tensor(targets[mask], dtype=torch.long)
    y_pred = (y_prob >= threshold).long()

    # Accuracy is always defined when the bin contains samples
    accuracy = (y_pred == y_true).float().mean().item()

    # Check whether the bin contains positive targets/predictions
    has_positive_targets = (y_true == 1).any().item()
    has_positive_predictions = (y_pred == 1).any().item()

    # Precision is undefined if there are no positive predictions
    precision = (
        binary_precision(y_pred, y_true).item()
        if has_positive_predictions
        else np.nan
    )

    # Recall is undefined if there are no positive targets
    recall = (
        binary_recall(y_pred, y_true).item()
        if has_positive_targets
        else np.nan
    )

    # F1 requires both positive targets and positive predictions
    f1 = (
        binary_f1_score(y_pred, y_true).item()
        if has_positive_targets and has_positive_predictions
        else np.nan
    )

    return np.array([precision, recall, f1, accuracy])


def plot_aggregate_uncertainty_performance(runs, save_path, bin_width=0.1):
    """
    Uncertainty analysis.
    Panel (a): performance within uncertainty intervals.
    Panel (b): cumulative performance up to each uncertainty threshold.
    """

    edges = np.arange(0, 1 + bin_width, bin_width)
    centres = (edges[:-1] + edges[1:]) / 2

    thresholds = edges[1:]

    interval_all = []
    cumulative_all = []
    interval_pct = []
    cumulative_pct = []

    for run in runs:
        probs = run["probabilities"][:, 1]
        targets = run["targets"].reshape(-1)
        uncertainty = run["uncertainties"].reshape(-1)
        decision_threshold = run["decision_threshold"].item()

        run_interval = []
        run_interval_pct = []

        for i in range(len(edges) - 1):
            mask = (
                (uncertainty >= edges[i])
                & (
                    uncertainty <= edges[i + 1]
                    if i == len(edges) - 2
                    else uncertainty < edges[i + 1]
                )
            )

            run_interval.append(
                _binary_metrics(
                    probs,
                    targets,
                    mask,
                    decision_threshold,
                )
            )

            run_interval_pct.append(100 * mask.mean())

        run_cumulative = []
        run_cumulative_pct = []

        for u in thresholds:
            mask = uncertainty <= u

            run_cumulative.append(
                _binary_metrics(
                    probs,
                    targets,
                    mask,
                    decision_threshold,
                )
            )

            run_cumulative_pct.append(100 * mask.mean())

        interval_all.append(run_interval)
        cumulative_all.append(run_cumulative)

        interval_pct.append(run_interval_pct)
        cumulative_pct.append(run_cumulative_pct)

    interval_all = np.asarray(interval_all)
    cumulative_all = np.asarray(cumulative_all)

    interval_mean = np.nanmean(interval_all, axis=0)
    cumulative_mean = np.nanmean(cumulative_all, axis=0)
    interval_sd = np.nanstd(interval_all, axis=0, ddof=1)
    cumulative_sd = np.nanstd(cumulative_all, axis=0, ddof=1)

    names = [
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.0))

    panels = [
        (
            axes[0],
            centres,
            interval_mean,
            interval_sd,
            np.mean(interval_pct, axis=0),
            "(a) Performance within uncertainty intervals",
            "Samples (%)",
        ),
        (
            axes[1],
            thresholds,
            cumulative_mean,
            cumulative_sd,
            np.mean(cumulative_pct, axis=0),
            "(b) Cumulative performance by uncertainty threshold",
            "Cumulative samples (%)",
        ),
    ]

    for ax, x, mean, sd, bars, title, bar_label in panels:
        bar_ax = ax.twinx()

        bar_ax.bar(
            x,
            bars,
            width=bin_width * 0.75,
            color= "#B0B0B0",
            alpha=0.35,
        )

        # Mean metric values across seeds
        for i, name in enumerate(names):
            ax.plot(
                x,
                mean[:, i],
                "o-",
                markersize=3,
                label=name,
            )

            # Between-seed variability
            ax.fill_between(
                x,
                np.clip(mean[:, i] - sd[:, i], 0, 1),
                np.clip(mean[:, i] + sd[:, i], 0, 1),
                alpha=0.06,
            )

        ax.set(
            title=title,
            ylabel="Predictive performance",
            ylim=(0, 1.05),
        )

        bar_ax.set_ylabel(bar_label)

    axes[0].set_xlabel("Evidential uncertainty")
    axes[1].set_xlabel("Evidential uncertainty threshold")
    
    # Remove grid from both panels
    ax.grid(False)
    bar_ax.grid(False)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.00),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_risk_coverage(runs, save_path):
    # Plot mean risk-coverage curve across seeds

    grid = np.linspace(0.05, 1.0, 100)
    interpolated_risks = []
    aurcs = []

    for run in runs:
        coverage = run["coverage"].reshape(-1)
        risk = run["risk"].reshape(-1)

        # Ensure coverage is sorted before interpolation
        order = np.argsort(coverage)
        coverage = coverage[order]
        risk = risk[order]

        # Remove duplicate coverage values
        coverage, unique_idx = np.unique(
            coverage,
            return_index=True,
        )
        risk = risk[unique_idx]

        # Interpolate each seed's curve onto a common grid
        interpolated_risks.append(
            np.interp(grid, coverage, risk)
        )

        aurcs.append(run["aurc"].item())

    interpolated_risks = np.stack(interpolated_risks)

    mean_risk = interpolated_risks.mean(axis=0)
    sd_risk = interpolated_risks.std(axis=0, ddof=1)

    mean_aurc = np.mean(aurcs)
    sd_aurc = np.std(aurcs, ddof=1)

    fig, ax = plt.subplots(figsize=(4.5, 3.4))

    ax.plot(
        grid,
        mean_risk,
        linewidth=1.8,
        color=COLOURS["edl_colour"],
    )

    ax.fill_between(
        grid,
        np.clip(mean_risk - sd_risk, 0, None),
        mean_risk + sd_risk,
        alpha=0.20,
    )

    # Show mean AURC across seeds
    ax.text(
        0.96,
        0.94,
        f"AURC = {mean_aurc:.4f} ± {sd_aurc:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    ax.set(
        xlabel="Coverage",
        ylabel="Prediction risk (1 - accuracy)",
        xlim=(0, 1),
        ylim=(0, None),
    )

    ax.grid(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_precision_recall(det_runs, edl_runs, save_path):
    # Plot mean precision-recall curves across seeds with +/- 1 SD

    # Common recall grid used to align curves from all seeds
    recall_grid = np.linspace(0, 1, 200)

    def curves(runs):
        # Store one interpolated precision-recall curve per seed
        values = []

        for run in runs:
            precision, recall, _ = precision_recall_curve(
                run["targets"].reshape(-1),
                run["probabilities"][:, 1],
            )

            # Sort recall values before interpolation
            order = np.argsort(recall)

            # Interpolate each seed onto the common recall grid
            values.append(
                np.interp(
                    recall_grid,
                    recall[order],
                    precision[order],
                )
            )

        values = np.stack(values)

        # Mean curve and between-seed standard deviation
        return (
            values.mean(axis=0),
            values.std(axis=0, ddof=1),
        )

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Plot deterministic and evidential models consistently
    for runs, label, color in [
        (det_runs, "Deterministic", COLOURS["det_colour"]),
        (edl_runs, "Evidential", COLOURS["edl_colour"]),
    ]:
        mean, sd = curves(runs)

        # Mean precision-recall curve
        ax.plot(
            recall_grid,
            mean,
            color=color,
            label=label,
        )

        # Shaded region shows +/- 1 SD across seeds
        ax.fill_between(
            recall_grid,
            np.clip(mean - sd, 0, 1),
            np.clip(mean + sd, 0, 1),
            color=color,
            alpha=0.15,
        )

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.legend(frameon=False)
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_confusion_matrices(det_runs, edl_runs, save_path):
    # Plot mean row-normalised confusion matrices across seeds

    def summary(runs):
        matrices = []

        for run in runs:
            cm = np.array([
                [run["tn"].item(), run["fp"].item()],
                [run["fn"].item(), run["tp"].item()],
            ], dtype=float)

            cm /= cm.sum(axis=1, keepdims=True)
            matrices.append(cm)

        matrices = np.stack(matrices)

        return (
            matrices.mean(axis=0),
            matrices.std(axis=0, ddof=1),
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3))
    labels = ["Non-crossing", "Crossing"]

    for ax, runs, title in [
        (axes[0], det_runs, "(a) Deterministic"),
        (axes[1], edl_runs, "(b) Evidential"),
    ]:
        mean, sd = summary(runs)

        # Mean percentage +/- 1 SD
        annotations = np.array([
            [
                f"{mean[r, c] * 100:.1f}%\n± {sd[r, c] * 100:.1f}"
                for c in range(2)
            ]
            for r in range(2)
        ])

        sns.heatmap(
            mean,
            annot=annotations,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )

        ax.set(
            title=title,
            xlabel="Predicted class",
            ylabel="Actual class",
        )

        ax.tick_params(rotation=0)
        ax.grid(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)