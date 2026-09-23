# Import libraries
import torch


def binary_brier_score(probabilities, targets) -> torch.Tensor:
    """
    Calculates the Binary Brier score. 
    The Brier score measures the mean squared difference between the
    predicted probability of the positive class and the observed binary
    outcome.
    Lower is better.
    """

    probabilities = probabilities.detach().reshape(-1).float()
    targets = targets.detach().reshape(-1).float()

    if probabilities.numel() != targets.numel():
        raise ValueError("probabilities and targets must have equal length") 

    return torch.mean((probabilities - targets) ** 2)


def expected_calibration_error(class_probabilities, targets, n_bins):
    """
    Calculates the Expected Calibration Error (ECE).
    ECE compares model confidence with empirical classification accuracy.
    Lower is better.
    """

    class_probabilities = class_probabilities.detach().float()
    targets = targets.detach().reshape(-1).long()

    if class_probabilities.ndim != 2:
        raise ValueError("class_probabilities must have shape [N, C].")

    if class_probabilities.shape[0] != targets.numel():
        raise ValueError("Number of predictions and targets must match.")

    confidence, predictions = torch.max(class_probabilities, dim=1)
    correct = (predictions == targets).float()

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        n_bins + 1,
        device=class_probabilities.device,
    )

    ece = torch.zeros((), device=class_probabilities.device)

    for bin_index in range(n_bins):
        lower = bin_boundaries[bin_index]
        upper = bin_boundaries[bin_index + 1]

        if bin_index == 0:
            in_bin = ((confidence >= lower) & (confidence <= upper))
        else:
            in_bin = ((confidence > lower) & (confidence <= upper))

        bin_count = in_bin.sum()
        if bin_count == 0:
            continue

        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confidence[in_bin].mean()
        bin_weight = bin_count.float() / targets.numel()

        ece += (bin_weight * torch.abs(bin_accuracy - bin_confidence))

    return ece


def risk_coverage_curve(uncertainty, predictions, targets):
    """
    Compute the risk-coverage curve.
    Samples are ordered from lowest to highest uncertainty.
    Coverage: Fraction of predictions retained.
    Risk: Classification error among retained predictions: risk = 1 - accuracy
    """

    uncertainty = uncertainty.detach().reshape(-1).float()
    predictions = predictions.detach().reshape(-1).long()
    targets = targets.detach().reshape(-1).long()
    n_samples = targets.numel()

    if (
        uncertainty.numel() != n_samples
        or predictions.numel() != n_samples
    ):
        raise ValueError("uncertainty, predictions and targets must have the same number of samples.")

    if n_samples == 0:
        raise ValueError("Cannot calculate a risk-coverage curve for an empty dataset.")

    # Rank samples from most certain to most uncertain.
    sorted_indices = torch.argsort(uncertainty, descending=False)

    sorted_uncertainty = uncertainty[sorted_indices]
    sorted_predictions = predictions[sorted_indices]
    sorted_targets = targets[sorted_indices]

    errors = (sorted_predictions != sorted_targets).float()
    cumulative_errors = torch.cumsum(errors, dim=0)

    retained_samples = torch.arange(
        1,
        n_samples + 1,
        device=uncertainty.device,
        dtype=torch.float64,
    )

    # Classification error rate among retained predictions.
    risk = cumulative_errors / retained_samples
    # Fraction of all samples retained.
    coverage = retained_samples / n_samples
    
    return coverage, risk, sorted_uncertainty


def area_under_risk_coverage_curve(coverage, risk):
    """
    Area Under the Risk-Coverage Curve (AURC).
    AURC summarises selective-prediction performance across all coverage levels. 
    Lower values indicate that the uncertainty ranking is better at
    retaining correct predictions while rejecting uncertain predictions.
    """

    coverage = coverage.detach().reshape(-1).float()
    risk = risk.detach().reshape(-1).float()

    if coverage.numel() != risk.numel():
        raise ValueError("coverage and risk must contain have equal length.")

    if coverage.numel() < 2:
        return torch.tensor(0.0, device=coverage.device)

    # Include the origin so the integration [0,1] 
    zero = torch.zeros(
        1,
        device=coverage.device,
        dtype=coverage.dtype,
    )

    coverage_for_auc = torch.cat([zero, coverage])
    risk_for_auc = torch.cat([zero, risk])

    # Trapezoidal numerical integration
    return torch.trapz(risk_for_auc, coverage_for_auc)


def calculate_uncertainty_metrics(class_probabilities, targets, uncertainty, n_bins, decision_threshold):

    # Function that calculates uncertainty and calibration metrics.
    # Brier, ECE, Coverage and Risk, AURC

    class_probabilities = class_probabilities.detach().float()
    targets = targets.detach().reshape(-1).long()
    uncertainty = uncertainty.detach().reshape(-1).float()
    
    # Probability assigned to the positive/crossing class.
    positive_probabilities = class_probabilities[:, 1]
    predictions = (positive_probabilities >= decision_threshold).long()

    # Brier Score
    brier = binary_brier_score(
        probabilities=positive_probabilities,
        targets=targets,
    )

    # ECE
    ece = expected_calibration_error(
        class_probabilities=class_probabilities,
        targets=targets,
        n_bins=n_bins,
    )

    coverage, risk, sorted_uncertainty = (
        risk_coverage_curve(
            uncertainty=uncertainty,
            predictions=predictions,
            targets=targets,
        )
    )
    
    # AURC
    aurc = area_under_risk_coverage_curve(coverage=coverage, risk=risk)

    return {
        "brier": brier,
        "ece": ece,
        "aurc": aurc,
        "coverage": coverage,
        "risk": risk,
        "sorted_uncertainty": sorted_uncertainty,
    }