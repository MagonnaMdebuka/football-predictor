"""Pass/fail gate checks for half-time model backtest results.

The HT and 2H models must each beat base-rate on RPS and log loss.
The HT/FT 9-outcome model must beat uniform 1/9.
A too-good alarm fires when model RPS is suspiciously far below baseline.
"""

from __future__ import annotations

import numpy as np

from services.engine.backtest.metrics import (
    log_loss,
    ranked_probability_score,
    hit_rate,
)
from services.engine.backtest.types import (
    GateDetail,
    HalfTimeMetricSummary,
    HalfTimePrediction,
)

# Model RPS below this fraction of baseline is suspiciously good
TOO_GOOD_RATIO = 0.9


def _compute_ht_metrics(preds: list[HalfTimePrediction]) -> HalfTimeMetricSummary:
    """Compute HT, 2H, and HT/FT metrics from predictions."""
    n = len(preds)

    # HT 1X2
    ht_actual = np.array([p.ht_result for p in preds])
    ht_p_home = np.array([p.ht_model_home for p in preds], dtype=np.float64)
    ht_p_draw = np.array([p.ht_model_draw for p in preds], dtype=np.float64)
    ht_p_away = np.array([p.ht_model_away for p in preds], dtype=np.float64)

    ht_rps = ranked_probability_score(ht_p_home, ht_p_draw, ht_p_away, ht_actual)
    ht_ll = log_loss(ht_p_home, ht_p_draw, ht_p_away, ht_actual)
    ht_hr = hit_rate(ht_p_home, ht_p_draw, ht_p_away, ht_actual)

    # 2H 1X2
    sh_actual = np.array([_sh_result(p) for p in preds])
    sh_p_home = np.array([p.sh_model_home for p in preds], dtype=np.float64)
    sh_p_draw = np.array([p.sh_model_draw for p in preds], dtype=np.float64)
    sh_p_away = np.array([p.sh_model_away for p in preds], dtype=np.float64)

    sh_rps = ranked_probability_score(sh_p_home, sh_p_draw, sh_p_away, sh_actual)
    sh_ll = log_loss(sh_p_home, sh_p_draw, sh_p_away, sh_actual)
    sh_hr = hit_rate(sh_p_home, sh_p_draw, sh_p_away, sh_actual)

    # HT/FT RPS (9-outcome)
    htft_rps = _compute_htft_rps(preds)

    return HalfTimeMetricSummary(
        ht_rps=round(ht_rps, 6),
        sh_rps=round(sh_rps, 6),
        htft_rps=round(htft_rps, 6),
        ht_log_loss=round(ht_ll, 6),
        sh_log_loss=round(sh_ll, 6),
        n_predictions=n,
        ht_hit_rate=round(ht_hr, 4),
        sh_hit_rate=round(sh_hr, 4),
    )


def _sh_result(p: HalfTimePrediction) -> str:
    """Compute 2H result from second-half goals."""
    if p.sh_home_goals > p.sh_away_goals:
        return "H"
    elif p.sh_home_goals == p.sh_away_goals:
        return "D"
    return "A"


def _compute_htft_rps(preds: list[HalfTimePrediction]) -> float:
    """Compute mean RPS for the 9-outcome HT/FT market.

    RPS for K ordered categories: (1/(K-1)) * sum_{k=1}^{K-1} (cum_p_k - cum_o_k)^2
    The 9 outcomes are ordered: HH, HD, HA, DH, DD, DA, AH, AD, AA.
    """
    outcomes = ["HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"]
    n = len(preds)
    if n == 0:
        return 0.0

    rps_sum = 0.0
    for p in preds:
        # Actual outcome
        actual_key = p.ht_result + p.ft_result
        # Predicted and actual cumulative distributions
        cum_p = 0.0
        cum_o = 0.0
        sq_sum = 0.0
        for i, outcome in enumerate(outcomes[:-1]):  # K-1 terms
            cum_p += p.htft_probs.get(outcome, 0.0)
            cum_o += 1.0 if outcomes[i] == actual_key else 0.0
            sq_sum += (cum_p - cum_o) ** 2
        rps_sum += sq_sum / (len(outcomes) - 1)

    return rps_sum / n


def _compute_ht_baselines(
    preds: list[HalfTimePrediction],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute HT and 2H base-rate proportions from predictions.

    Returns:
        ((ht_home, ht_draw, ht_away), (sh_home, sh_draw, sh_away))
    """
    n = len(preds)
    ht_h = sum(1 for p in preds if p.ht_result == "H") / n
    ht_d = sum(1 for p in preds if p.ht_result == "D") / n
    ht_a = sum(1 for p in preds if p.ht_result == "A") / n

    sh_h = sum(1 for p in preds if _sh_result(p) == "H") / n
    sh_d = sum(1 for p in preds if _sh_result(p) == "D") / n
    sh_a = sum(1 for p in preds if _sh_result(p) == "A") / n

    return (ht_h, ht_d, ht_a), (sh_h, sh_d, sh_a)


def run_halftime_gate_checks(
    preds: list[HalfTimePrediction],
) -> tuple[HalfTimeMetricSummary | None, bool, list[GateDetail]]:
    """Run all gate checks for the half-time model.

    Returns:
        (metrics, gate_passed, gate_details)
    """
    if not preds:
        detail = GateDetail(
            name="halftime_no_predictions",
            passed=True,
            message="No half-time predictions — gate skipped",
        )
        return None, True, [detail]

    metrics = _compute_ht_metrics(preds)
    details: list[GateDetail] = []

    # Compute baselines
    ht_br, sh_br = _compute_ht_baselines(preds)
    n = len(preds)

    # HT base-rate RPS
    ht_actual = np.array([p.ht_result for p in preds])
    ht_br_rps = ranked_probability_score(
        np.full(n, ht_br[0]), np.full(n, ht_br[1]), np.full(n, ht_br[2]),
        ht_actual,
    )
    details.append(GateDetail(
        name="ht_rps_vs_baseline",
        passed=metrics.ht_rps < ht_br_rps,
        message=f"HT model RPS={metrics.ht_rps:.6f} vs baseline={ht_br_rps:.6f}",
    ))

    # 2H base-rate RPS
    sh_actual = np.array([_sh_result(p) for p in preds])
    sh_br_rps = ranked_probability_score(
        np.full(n, sh_br[0]), np.full(n, sh_br[1]), np.full(n, sh_br[2]),
        sh_actual,
    )
    details.append(GateDetail(
        name="sh_rps_vs_baseline",
        passed=metrics.sh_rps < sh_br_rps,
        message=f"2H model RPS={metrics.sh_rps:.6f} vs baseline={sh_br_rps:.6f}",
    ))

    # HT/FT RPS vs uniform 1/9
    uniform_htft_rps = _compute_uniform_htft_rps(preds)
    details.append(GateDetail(
        name="htft_rps_vs_uniform",
        passed=metrics.htft_rps < uniform_htft_rps,
        message=f"HT/FT RPS={metrics.htft_rps:.6f} vs uniform={uniform_htft_rps:.6f}",
    ))

    # HT log loss vs baseline
    ht_br_ll = log_loss(
        np.full(n, ht_br[0]), np.full(n, ht_br[1]), np.full(n, ht_br[2]),
        ht_actual,
    )
    details.append(GateDetail(
        name="ht_log_loss_vs_baseline",
        passed=metrics.ht_log_loss < ht_br_ll,
        message=f"HT log_loss={metrics.ht_log_loss:.6f} vs baseline={ht_br_ll:.6f}",
    ))

    # 2H log loss vs baseline
    sh_br_ll = log_loss(
        np.full(n, sh_br[0]), np.full(n, sh_br[1]), np.full(n, sh_br[2]),
        sh_actual,
    )
    details.append(GateDetail(
        name="sh_log_loss_vs_baseline",
        passed=metrics.sh_log_loss < sh_br_ll,
        message=f"2H log_loss={metrics.sh_log_loss:.6f} vs baseline={sh_br_ll:.6f}",
    ))

    # Too-good alarm: HT RPS suspiciously low
    if ht_br_rps > 0:
        ratio = metrics.ht_rps / ht_br_rps
        too_good = ratio < TOO_GOOD_RATIO
        details.append(GateDetail(
            name="ht_too_good",
            passed=not too_good,
            message=(
                f"HT model/baseline ratio={ratio:.4f} "
                f"{'<' if too_good else '>='} threshold={TOO_GOOD_RATIO} — "
                f"{'suspected leakage' if too_good else 'OK'}"
            ),
        ))

    gate_passed = all(d.passed for d in details)
    return metrics, gate_passed, details


def _compute_uniform_htft_rps(preds: list[HalfTimePrediction]) -> float:
    """Compute mean RPS for uniform 1/9 prediction across 9 HT/FT outcomes."""
    outcomes = ["HH", "HD", "HA", "DH", "DD", "DA", "AH", "AD", "AA"]
    n = len(preds)
    if n == 0:
        return 0.0

    rps_sum = 0.0
    for p in preds:
        actual_key = p.ht_result + p.ft_result
        cum_p = 0.0
        cum_o = 0.0
        sq_sum = 0.0
        for i, outcome in enumerate(outcomes[:-1]):
            cum_p += 1.0 / 9.0
            cum_o += 1.0 if outcomes[i] == actual_key else 0.0
            sq_sum += (cum_p - cum_o) ** 2
        rps_sum += sq_sum / (len(outcomes) - 1)

    return rps_sum / n
