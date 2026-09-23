"""Pass/fail gate checks for backtest results.

The model must beat base-rate AND independent Poisson on both RPS and
log loss. Bookmaker is a reference ceiling only (not a gate requirement).

A too-good alarm fires when results are suspiciously strong, suggesting
possible data leakage:
- Combined RPS < 0.190
- Model beats bookmaker on log loss
"""

from __future__ import annotations

from services.engine.backtest.types import GateDetail, MetricSet, SeasonMetrics

# Threshold below which RPS is suspiciously good
TOO_GOOD_RPS_THRESHOLD = 0.190


def check_beats_baseline(
    model: MetricSet,
    baseline: MetricSet,
    baseline_name: str,
    metric_name: str,
) -> GateDetail:
    """Check that model beats a baseline on a specific metric (lower is better).

    Args:
        model: model metric set
        baseline: baseline metric set
        baseline_name: human-readable baseline name
        metric_name: 'rps' or 'log_loss'

    Returns:
        GateDetail with pass/fail result.
    """
    model_val = getattr(model, metric_name)
    baseline_val = getattr(baseline, metric_name)
    passed = model_val < baseline_val
    message = (
        f"model {metric_name}={model_val:.6f} vs "
        f"{baseline_name} {metric_name}={baseline_val:.6f}"
    )
    return GateDetail(
        name=f"{metric_name}_vs_{baseline_name}",
        passed=passed,
        message=message,
    )


def check_too_good_rps(model: MetricSet) -> GateDetail:
    """Check that combined RPS is not suspiciously low (potential leakage).

    Args:
        model: model metric set

    Returns:
        GateDetail — passed=True means NOT too good (healthy).
    """
    too_good = model.rps < TOO_GOOD_RPS_THRESHOLD
    return GateDetail(
        name="too_good_rps",
        passed=not too_good,
        message=(
            f"model rps={model.rps:.6f} {'<' if too_good else '>='} "
            f"threshold={TOO_GOOD_RPS_THRESHOLD} — "
            f"{'suspected leakage' if too_good else 'OK'}"
        ),
    )


def check_too_good_vs_bookmaker(
    model: MetricSet,
    bookmaker: MetricSet | None,
) -> GateDetail:
    """Check that model does not beat bookmaker on log loss (potential leakage).

    Args:
        model: model metric set
        bookmaker: bookmaker metric set (None if unavailable)

    Returns:
        GateDetail — passed=True means NOT too good (healthy).
    """
    if bookmaker is None:
        return GateDetail(
            name="too_good_vs_bookmaker",
            passed=True,
            message="bookmaker baseline unavailable — check skipped",
        )

    too_good = model.log_loss < bookmaker.log_loss
    return GateDetail(
        name="too_good_vs_bookmaker",
        passed=not too_good,
        message=(
            f"model log_loss={model.log_loss:.6f} vs bookmaker log_loss={bookmaker.log_loss:.6f}"
            f" — {'suspected leakage' if too_good else 'OK'}"
        ),
    )


def run_gate_checks(combined: SeasonMetrics) -> tuple[bool, list[GateDetail]]:
    """Run all gate checks on combined metrics.

    Gate requirements:
    - Model beats base_rate on RPS
    - Model beats base_rate on log_loss
    - Model beats independent_poisson on RPS
    - Model beats independent_poisson on log_loss
    - RPS not suspiciously low (too-good alarm)
    - Model does not beat bookmaker on log loss (too-good alarm)

    Args:
        combined: combined SeasonMetrics across all held-out seasons

    Returns:
        (gate_passed, gate_details) where gate_passed is True only if
        all checks pass.
    """
    details: list[GateDetail] = []

    # Gate: model must beat base_rate
    details.append(check_beats_baseline(
        combined.model, combined.base_rate, "base_rate", "rps",
    ))
    details.append(check_beats_baseline(
        combined.model, combined.base_rate, "base_rate", "log_loss",
    ))

    # Gate: model must beat independent_poisson
    details.append(check_beats_baseline(
        combined.model, combined.independent_poisson, "independent_poisson", "rps",
    ))
    details.append(check_beats_baseline(
        combined.model, combined.independent_poisson, "independent_poisson", "log_loss",
    ))

    # Too-good alarms
    details.append(check_too_good_rps(combined.model))
    details.append(check_too_good_vs_bookmaker(combined.model, combined.bookmaker))

    gate_passed = all(d.passed for d in details)
    return gate_passed, details
