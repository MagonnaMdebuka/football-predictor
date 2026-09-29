"""Reliability diagram data computation."""

from __future__ import annotations

import numpy as np


def compute_reliability_data(
    predicted: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
    min_bin_count: int = 30,
) -> dict:
    """Compute reliability diagram data points.

    Parameters
    ----------
    predicted : array of predicted probabilities
    observed : array of binary outcomes (0 or 1)
    n_bins : number of equal-width bins
    min_bin_count : bins with fewer observations are flagged as
        ``insufficient_data`` and excluded from calibration_error

    Returns
    -------
    Dict with keys:
        bins: list of {midpoint, predicted_mean, observed_mean, count,
              insufficient_data}
        calibration_error: max |pred - obs| across sufficiently-populated bins
        mean_calibration_error: weighted mean |pred - obs|
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    total_count = 0
    weighted_error_sum = 0.0
    max_error = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        midpoint = (lo + hi) / 2

        if i < n_bins - 1:
            mask = (predicted >= lo) & (predicted < hi)
        else:
            mask = (predicted >= lo) & (predicted <= hi)

        count = int(mask.sum())
        insufficient = count < min_bin_count

        if count > 0:
            pred_mean = float(predicted[mask].mean())
            obs_mean = float(observed[mask].mean())
            error = abs(pred_mean - obs_mean)
            # Only include sufficiently-populated bins in error metrics
            if not insufficient:
                max_error = max(max_error, error)
                weighted_error_sum += error * count
                total_count += count
        else:
            pred_mean = float(midpoint)
            obs_mean = 0.0

        bins.append({
            "midpoint": float(midpoint),
            "predicted_mean": pred_mean,
            "observed_mean": obs_mean,
            "count": count,
            "insufficient_data": insufficient,
        })

    mce = weighted_error_sum / total_count if total_count > 0 else 0.0

    return {
        "bins": bins,
        "calibration_error": max_error,
        "mean_calibration_error": mce,
    }
