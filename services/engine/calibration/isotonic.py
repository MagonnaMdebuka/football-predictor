"""Pure-numpy isotonic regression via pool adjacent violators (PAVA).

No scikit-learn dependency — implements PAVA directly.
"""

from __future__ import annotations

import numpy as np


def isotonic_regression(
    y: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Pool adjacent violators — monotone non-decreasing fit.

    Parameters
    ----------
    y : array of shape (n,)
        Target values.
    weights : array of shape (n,) or None
        Sample weights. Uniform if None.

    Returns
    -------
    array of shape (n,)
        Fitted values satisfying y_fit[0] <= y_fit[1] <= ... <= y_fit[n-1].
    """
    n = len(y)
    if n == 0:
        return np.array([], dtype=float)

    y = np.asarray(y, dtype=float)
    if weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    # Each block is [weighted_sum, total_weight]
    blocks: list[list[float]] = []
    block_ranges: list[list[int]] = []  # [start, end) indices

    for i in range(n):
        blocks.append([y[i] * weights[i], weights[i]])
        block_ranges.append([i, i + 1])

        # Merge with previous block while monotonicity is violated
        while len(blocks) > 1:
            cur_mean = blocks[-1][0] / blocks[-1][1]
            prev_mean = blocks[-2][0] / blocks[-2][1]
            if prev_mean <= cur_mean:
                break
            # Merge current into previous
            blocks[-2][0] += blocks[-1][0]
            blocks[-2][1] += blocks[-1][1]
            block_ranges[-2][1] = block_ranges[-1][1]
            blocks.pop()
            block_ranges.pop()

    result = np.empty(n, dtype=float)
    for block, (start, end) in zip(blocks, block_ranges):
        result[start:end] = block[0] / block[1]

    return result


def fit_calibration_bins(
    predicted: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Fit isotonic regression and return calibration bins.

    Parameters
    ----------
    predicted : array of predicted probabilities
    observed : array of binary outcomes (0 or 1)
    n_bins : number of equal-width bins across [0, 1]

    Returns
    -------
    List of dicts with keys: bin_lower, bin_upper, predicted_frequency,
    observed_frequency, sample_size.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []

    # Compute per-bin means
    bin_pred_means = []
    bin_obs_means = []
    bin_sizes = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (predicted >= lo) & (predicted < hi)
        else:
            # Last bin includes right edge
            mask = (predicted >= lo) & (predicted <= hi)

        count = int(mask.sum())
        if count > 0:
            pred_mean = float(predicted[mask].mean())
            obs_mean = float(observed[mask].mean())
        else:
            pred_mean = float((lo + hi) / 2)
            obs_mean = 0.0

        bin_pred_means.append(pred_mean)
        bin_obs_means.append(obs_mean)
        bin_sizes.append(count)

    # Apply PAVA to observed means (weighted by sample sizes)
    obs_arr = np.array(bin_obs_means)
    weight_arr = np.array(bin_sizes, dtype=float)
    # Only apply PAVA to bins with data; leave empty bins as-is
    has_data = weight_arr > 0
    if has_data.sum() > 0:
        calibrated = obs_arr.copy()
        data_indices = np.where(has_data)[0]
        fitted = isotonic_regression(
            obs_arr[data_indices], weight_arr[data_indices]
        )
        calibrated[data_indices] = fitted
    else:
        calibrated = obs_arr

    for i in range(n_bins):
        bins.append({
            "bin_lower": float(bin_edges[i]),
            "bin_upper": float(bin_edges[i + 1]),
            "predicted_frequency": bin_pred_means[i],
            "observed_frequency": float(calibrated[i]),
            "sample_size": bin_sizes[i],
        })

    return bins
