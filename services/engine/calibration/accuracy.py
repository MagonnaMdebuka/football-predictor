"""Scoring metrics for calibration and accuracy assessment."""

from __future__ import annotations

import numpy as np


def brier_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Brier score for binary outcomes.

    Lower is better. Perfect = 0, uninformative on balanced binary = 0.25.
    """
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    return float(np.mean((predicted - observed) ** 2))


def log_loss(predicted: np.ndarray, observed: np.ndarray, eps: float = 1e-15) -> float:
    """Log loss (binary cross-entropy) for binary outcomes."""
    predicted = np.clip(np.asarray(predicted, dtype=float), eps, 1.0 - eps)
    observed = np.asarray(observed, dtype=float)
    return float(-np.mean(observed * np.log(predicted) + (1 - observed) * np.log(1 - predicted)))


def rps_score(probs: np.ndarray, outcome_idx: int) -> float:
    """Ranked probability score for a single multi-outcome prediction.

    Parameters
    ----------
    probs : array of probabilities (e.g. [home, draw, away] for 1X2)
    outcome_idx : index of the actual outcome (0, 1, or 2)

    Returns
    -------
    RPS value. Lower is better.
    """
    probs = np.asarray(probs, dtype=float)
    n = len(probs)
    actual = np.zeros(n)
    actual[outcome_idx] = 1.0

    cum_pred = np.cumsum(probs)
    cum_actual = np.cumsum(actual)
    return float(np.mean((cum_pred - cum_actual) ** 2))


def hit_rate(predicted_classes: np.ndarray, actual_classes: np.ndarray) -> float:
    """Fraction of correct predictions."""
    predicted_classes = np.asarray(predicted_classes)
    actual_classes = np.asarray(actual_classes)
    if len(predicted_classes) == 0:
        return 0.0
    return float(np.mean(predicted_classes == actual_classes))


def compute_accuracy_metrics(
    predictions: list[dict],
    market: str,
) -> dict:
    """Compute Brier, log loss, hit rate, RPS, sample size for a market.

    Parameters
    ----------
    predictions : list of dicts
        For binary markets: keys 'predicted_prob', 'observed' (0/1).
        For 1X2: keys 'probs' (3-element list), 'outcome_idx' (0/1/2).
    market : market identifier

    Returns
    -------
    Dict with keys: brier, log_loss, hit_rate, rps, sample_size.
    """
    n = len(predictions)
    if n == 0:
        return {
            "brier": None,
            "log_loss": None,
            "hit_rate": None,
            "rps": None,
            "sample_size": 0,
        }

    is_1x2 = "probs" in predictions[0]

    if is_1x2:
        rps_vals = [
            rps_score(np.array(p["probs"]), p["outcome_idx"]) for p in predictions
        ]
        # For hit rate: predicted class = argmax of probs
        pred_classes = np.array([np.argmax(p["probs"]) for p in predictions])
        actual_classes = np.array([p["outcome_idx"] for p in predictions])
        hr = hit_rate(pred_classes, actual_classes)

        return {
            "brier": None,
            "log_loss": None,
            "hit_rate": float(hr),
            "rps": float(np.mean(rps_vals)),
            "sample_size": n,
        }
    else:
        pred = np.array([p["predicted_prob"] for p in predictions])
        obs = np.array([p["observed"] for p in predictions])
        bs = brier_score(pred, obs)
        ll = log_loss(pred, obs)
        # Hit rate: predicted > 0.5 → 1, else 0
        pred_classes = (pred >= 0.5).astype(int)
        hr = hit_rate(pred_classes, obs.astype(int))

        return {
            "brier": float(bs),
            "log_loss": float(ll),
            "hit_rate": float(hr),
            "rps": None,
            "sample_size": n,
        }
