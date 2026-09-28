"""Confidence bucket classification from 1X2 probabilities.

Entropy of a 3-outcome distribution ranges from 0 (certainty) to
log2(3) ~ 1.585 (uniform). Thresholds map entropy to human-readable
confidence labels.
"""

from __future__ import annotations

import math

CONFIDENCE_THRESHOLDS: list[tuple[str, float, float]] = [
    ("very_high", 0.0, 1.0),
    ("high", 1.0, 1.35),
    ("medium", 1.35, 1.50),
    ("low", 1.50, 1.5850),
]


def compute_confidence(home: float, draw: float, away: float) -> str:
    """Classify a 1X2 distribution into a confidence bucket.

    Args:
        home: P(home win)
        draw: P(draw)
        away: P(away win)

    Returns:
        One of 'very_high', 'high', 'medium', 'low'.

    Raises:
        ValueError: if probabilities do not sum to approximately 1.0.
    """
    total = home + draw + away
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Probabilities must sum to 1.0, got {total:.4f}")

    entropy = 0.0
    for p in (home, draw, away):
        if p > 0:
            entropy -= p * math.log2(p)

    for label, lo, hi in CONFIDENCE_THRESHOLDS:
        if lo <= entropy < hi:
            return label

    return "low"
