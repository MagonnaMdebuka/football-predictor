"""Pass/fail gate checks for count model backtest results.

The count model must beat a constant-lambda Poisson baseline on mean Brier
score across publishable O/U lines. A too-good alarm fires when the model
Brier is suspiciously far below the baseline (ratio < 0.9), suggesting
data leakage.

The constant-lambda baseline uses the training-set mean count as lambda for
every match — the simplest possible model. The fitted NB2 model must outperform
this to demonstrate that team-level and (for cards) referee effects add value.

Gate evaluation uses a subset of publishable lines (CORNER_GATE_LINES,
BOOKING_GATE_LINES) — extreme lines near 0/1 carry no information and
inflate mean Brier noise.
"""

from __future__ import annotations

from services.engine.backtest.count_types import CountMetricSet, CountPrediction
from services.engine.backtest.types import GateDetail
from services.engine.markets.counts import (
    BOOKING_GATE_LINES,
    CORNER_GATE_LINES,
)


# Model Brier below this fraction of baseline Brier is suspiciously good
TOO_GOOD_RATIO = 0.9


def _gate_lines_for_model(model_type: str) -> list[float]:
    """Return the publishable gate lines for a given model type."""
    if model_type == "corners":
        return CORNER_GATE_LINES
    return BOOKING_GATE_LINES


def _compute_baseline_brier(
    predictions: list[CountPrediction],
    lines: list[float],
) -> float:
    """Compute mean Brier for a constant-lambda Poisson baseline.

    The baseline uses the mean observed total as its constant prediction for
    the "over" probability at each line. This is a simpler (but fair) baseline:
    P(over L) = fraction of training matches with total > L.

    For gate purposes we use the held-out actuals themselves as the reference
    distribution — the baseline simply predicts the overall over-rate observed
    in the held-out set for each line.
    """
    if not predictions or not lines:
        return 0.0

    actuals = [p.actual_home + p.actual_away for p in predictions]
    n = len(actuals)

    per_line_brier: list[float] = []
    for line in lines:
        # Baseline predicts P(over) = observed fraction of totals > line
        baseline_over = sum(1 for a in actuals if a > line) / n
        brier_sum = 0.0
        for a in actuals:
            indicator = 1.0 if a > line else 0.0
            brier_sum += (baseline_over - indicator) ** 2
        per_line_brier.append(brier_sum / n)

    return sum(per_line_brier) / len(per_line_brier)


def _compute_model_gate_brier(
    model_metrics: CountMetricSet,
    gate_lines: list[float],
) -> float:
    """Compute the model's mean Brier restricted to gate lines only.

    Rather than using model_metrics.mean_brier (which averages across ALL
    lines including extremes), average only the per-line Brier scores for
    the gate lines.
    """
    brier_vals = [
        model_metrics.per_line_brier[line]
        for line in gate_lines
        if line in model_metrics.per_line_brier
    ]
    if not brier_vals:
        return model_metrics.mean_brier
    return sum(brier_vals) / len(brier_vals)


def check_beats_baseline_brier(
    model_brier: float,
    baseline_brier: float,
    model_type: str,
) -> GateDetail:
    """Check that the model beats constant-lambda baseline on mean Brier.

    Args:
        model_brier: model's gate-filtered mean Brier score
        baseline_brier: baseline mean Brier score
        model_type: 'corners' or 'cards'

    Returns:
        GateDetail with pass/fail result.
    """
    passed = model_brier < baseline_brier
    return GateDetail(
        name=f"{model_type}_brier_vs_baseline",
        passed=passed,
        message=(
            f"{model_type} model mean_brier={model_brier:.6f} vs "
            f"baseline mean_brier={baseline_brier:.6f}"
        ),
    )


def check_too_good_brier(
    model_brier: float,
    baseline_brier: float,
    model_type: str,
) -> GateDetail:
    """Check that Brier score is not suspiciously low (potential leakage).

    Alarm fires when model_brier < TOO_GOOD_RATIO * baseline_brier.
    This relative check avoids false alarms on lopsided markets where
    both model and baseline Brier are legitimately low.

    Args:
        model_brier: model's gate-filtered mean Brier score
        baseline_brier: baseline mean Brier score
        model_type: 'corners' or 'cards'

    Returns:
        GateDetail — passed=True means NOT too good (healthy).
    """
    if baseline_brier <= 0:
        return GateDetail(
            name=f"{model_type}_too_good_brier",
            passed=True,
            message=f"{model_type} baseline_brier=0 — too-good check skipped",
        )

    ratio = model_brier / baseline_brier
    too_good = ratio < TOO_GOOD_RATIO
    return GateDetail(
        name=f"{model_type}_too_good_brier",
        passed=not too_good,
        message=(
            f"{model_type} model/baseline ratio={ratio:.4f} "
            f"{'<' if too_good else '>='} threshold={TOO_GOOD_RATIO} — "
            f"{'suspected leakage' if too_good else 'OK'}"
        ),
    )


def run_count_gate_checks(
    predictions: list[CountPrediction],
    model_metrics: CountMetricSet,
    model_type: str,
) -> tuple[bool, list[GateDetail]]:
    """Run all gate checks for a count model.

    Gate requirements:
    - Model beats constant-lambda baseline on mean Brier (gate lines only)
    - Brier not suspiciously low relative to baseline (too-good alarm)

    Args:
        predictions: list of count predictions (used to compute baseline)
        model_metrics: model's computed metrics
        model_type: 'corners' or 'cards'

    Returns:
        (gate_passed, gate_details)
    """
    details: list[GateDetail] = []

    if not predictions or model_metrics.n_predictions == 0:
        detail = GateDetail(
            name=f"{model_type}_no_predictions",
            passed=True,
            message=f"No {model_type} predictions — gate skipped",
        )
        return True, [detail]

    # Use gate lines (publishable subset) instead of all prediction lines
    gate_lines = _gate_lines_for_model(model_type)

    # Baseline Brier on gate lines only
    baseline_brier = _compute_baseline_brier(predictions, gate_lines)

    # Model Brier on gate lines only
    model_brier = _compute_model_gate_brier(model_metrics, gate_lines)

    # Gate: model must beat baseline
    details.append(check_beats_baseline_brier(model_brier, baseline_brier, model_type))

    # Too-good alarm (relative to baseline)
    details.append(check_too_good_brier(model_brier, baseline_brier, model_type))

    gate_passed = all(d.passed for d in details)
    return gate_passed, details
