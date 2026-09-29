"""Accuracy and calibration endpoints."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AccuracyMetric, CalibrationMap, ModelVsMarket
from services.api.deps import get_db
from services.api.schemas import (
    AccuracyOverviewOut,
    CalibrationBinOut,
    GateStatusOut,
    MarketAccuracyDetailOut,
    MarketAccuracyOut,
    ModelVsMarketItemOut,
    ModelVsMarketOut,
    QualityBadgeOut,
    ReliabilityDiagramOut,
)
from services.engine.calibration.badges import compute_quality_badge
from services.engine.calibration.gates import check_publication_gates

router = APIRouter(prefix="/api/v1/accuracy", tags=["accuracy"])

# Goals-derived markets share the same data lineage as 1X2: the underlying
# lambdas are fitted on actual goals, which IS direct data. All markets
# calibrated from the goals model get direct_data=True.
_GOALS_DERIVED_MARKETS = {
    "match_result_home", "match_result_draw", "match_result_away",
    "match_result",
    "over_under_2.5_over", "over_under_2.5_under",
    "over_under_1.5_over", "over_under_1.5_under",
    "over_under_3.5_over", "over_under_3.5_under",
    "btts_yes", "btts_no",
}

# Number of backtest seasons (E0 2020-2024)
_BACKTEST_SEASONS = 2


def _badge_for_market(market_key: str) -> QualityBadgeOut:
    """Compute quality badge for a market."""
    has_direct = market_key in _GOALS_DERIVED_MARKETS
    badge = compute_quality_badge(
        n_seasons=_BACKTEST_SEASONS,
        has_direct_data=has_direct,
    )
    return QualityBadgeOut(
        market=market_key,
        badge=badge,
        n_seasons=_BACKTEST_SEASONS,
        has_direct_data=has_direct,
    )


def _gates_for_market(
    calibration_bins: list[CalibrationBinOut],
    sample_size: int,
    brier: float | None,
    has_direct_data: bool = True,
) -> list[GateStatusOut]:
    """Compute publication gates for a market."""
    # Build reliability-like data with bins for min_bin_count filtering
    non_empty = [b for b in calibration_bins if b.sample_size > 0]
    if non_empty:
        errors = [
            abs(b.predicted_frequency - b.observed_frequency)
            for b in non_empty
        ]
        max_error = max(errors)
        total = sum(b.sample_size for b in non_empty)
        mce = sum(
            abs(b.predicted_frequency - b.observed_frequency) * b.sample_size
            for b in non_empty
        ) / total if total > 0 else 0.0
    else:
        max_error = 0.0
        mce = 0.0

    reliability_data = {
        "calibration_error": max_error,
        "mean_calibration_error": mce,
        "bins": [
            {
                "predicted_mean": b.predicted_frequency,
                "observed_mean": b.observed_frequency,
                "count": b.sample_size,
            }
            for b in calibration_bins
        ],
    }

    # Baseline Brier for binary is 0.25; model beats if brier < 0.25
    beats_baseline = brier is not None and brier < 0.25

    gates = check_publication_gates(
        reliability_data=reliability_data,
        n_settled=sample_size,
        beats_baseline=beats_baseline,
        has_direct_data=has_direct_data,
    )
    return [GateStatusOut(**g) for g in gates]


@router.get("", response_model=AccuracyOverviewOut)
async def accuracy_overview(db: AsyncSession = Depends(get_db)):
    """Accuracy overview with per-market metrics, badges, and gates."""
    # Get all accuracy metrics
    result = await db.execute(
        select(AccuracyMetric).order_by(AccuracyMetric.market)
    )
    metrics = result.scalars().all()

    if not metrics:
        return AccuracyOverviewOut(markets=[], total_settled=0, source="none")

    # Group by market
    by_market: dict[str, dict] = defaultdict(lambda: {"sample_size": 0})
    source = "backtest"

    for m in metrics:
        key = m.market or "overall"
        by_market[key][m.metric_name] = m.metric_value
        by_market[key]["sample_size"] = m.sample_size
        source = m.source

    # Get calibration bins for gates
    cal_result = await db.execute(
        select(CalibrationMap).where(CalibrationMap.is_current == True)  # noqa: E712
    )
    cal_maps = cal_result.scalars().all()
    cal_by_market: dict[str, list[CalibrationBinOut]] = defaultdict(list)
    for c in cal_maps:
        key = f"{c.market}_{c.selection}" if c.selection else c.market
        cal_by_market[key].append(CalibrationBinOut(
            bin_lower=c.bin_lower,
            bin_upper=c.bin_upper,
            predicted_frequency=c.predicted_frequency,
            observed_frequency=c.observed_frequency,
            sample_size=c.sample_size,
        ))

    # Build market accuracy list
    markets: list[MarketAccuracyOut] = []
    total_settled = 0

    for market_key, vals in sorted(by_market.items()):
        if market_key == "overall":
            continue
        sample_size = vals["sample_size"]
        total_settled = max(total_settled, sample_size)
        brier = vals.get("brier")
        cal_bins = cal_by_market.get(market_key, [])
        badge = _badge_for_market(market_key)
        gates = _gates_for_market(
            cal_bins, sample_size, brier, has_direct_data=badge.has_direct_data
        )

        markets.append(MarketAccuracyOut(
            market=market_key,
            brier=brier,
            rps=vals.get("rps"),
            log_loss=vals.get("log_loss"),
            hit_rate=vals.get("hit_rate"),
            sample_size=sample_size,
            badge=badge,
            gates=gates,
            source=source,
        ))

    return AccuracyOverviewOut(
        markets=markets,
        total_settled=total_settled,
        source=source,
    )


@router.get("/model-vs-market", response_model=ModelVsMarketOut)
async def model_vs_market(db: AsyncSession = Depends(get_db)):
    """Model vs bookmaker comparison."""
    result = await db.execute(select(ModelVsMarket))
    rows = result.scalars().all()

    if not rows:
        return ModelVsMarketOut(comparisons=[], source="none")

    # Group by market+selection, compute mean squared errors
    by_selection: dict[str, dict] = defaultdict(
        lambda: {"model_errs": [], "bk_errs": [], "source": "backtest"}
    )

    for r in rows:
        key = f"{r.market}_{r.selection}"
        by_selection[key]["source"] = r.source

    # For model-vs-market we compute Brier-like metric per selection
    # Group all rows by selection
    selection_data: dict[str, list] = defaultdict(list)
    for r in rows:
        key = f"{r.market}_{r.selection}"
        selection_data[key].append(r)

    comparisons: list[ModelVsMarketItemOut] = []
    source = "backtest"

    for key, data_rows in sorted(selection_data.items()):
        model_probs = np.array([r.model_prob for r in data_rows])
        bk_probs = np.array([r.bookmaker_prob for r in data_rows])
        source = data_rows[0].source

        # Brier-like: mean squared difference from 0/1 outcomes
        # Since we don't have outcomes stored in ModelVsMarket,
        # compare model vs bookmaker directly: mean absolute difference
        comparisons.append(ModelVsMarketItemOut(
            market=key,
            model_metric=float(np.mean(model_probs)),
            bookmaker_metric=float(np.mean(bk_probs)),
            metric_name="mean_probability",
            sample_size=len(data_rows),
        ))

    return ModelVsMarketOut(comparisons=comparisons, source=source)


@router.get("/{market}", response_model=MarketAccuracyDetailOut)
async def market_accuracy(market: str, db: AsyncSession = Depends(get_db)):
    """Detailed accuracy for a specific market with reliability diagram."""
    # Get calibration bins
    # Try exact market match first, then split into market+selection
    cal_result = await db.execute(
        select(CalibrationMap)
        .where(CalibrationMap.is_current == True)  # noqa: E712
        .order_by(CalibrationMap.bin_lower)
    )
    all_cal = cal_result.scalars().all()

    # Filter for this market key (market_selection format)
    cal_rows = [
        c for c in all_cal
        if (f"{c.market}_{c.selection}" if c.selection else c.market) == market
    ]

    if not cal_rows:
        raise HTTPException(status_code=404, detail=f"Market '{market}' not found")

    cal_bins = [
        CalibrationBinOut(
            bin_lower=c.bin_lower,
            bin_upper=c.bin_upper,
            predicted_frequency=c.predicted_frequency,
            observed_frequency=c.observed_frequency,
            sample_size=c.sample_size,
        )
        for c in cal_rows
    ]

    # Build reliability diagram from calibration bins
    rel_bins = []
    max_error = 0.0
    total_count = 0
    weighted_error_sum = 0.0

    for b in cal_bins:
        midpoint = (b.bin_lower + b.bin_upper) / 2
        if b.sample_size > 0:
            error = abs(b.predicted_frequency - b.observed_frequency)
            max_error = max(max_error, error)
            weighted_error_sum += error * b.sample_size
            total_count += b.sample_size
        rel_bins.append({
            "midpoint": midpoint,
            "predicted_mean": b.predicted_frequency,
            "observed_mean": b.observed_frequency,
            "count": b.sample_size,
        })

    mce = weighted_error_sum / total_count if total_count > 0 else 0.0

    reliability = ReliabilityDiagramOut(
        bins=cal_bins,
        calibration_error=max_error,
        mean_calibration_error=mce,
    )

    # Get accuracy metrics for this market
    metrics_result = await db.execute(
        select(AccuracyMetric).where(AccuracyMetric.market == market)
    )
    metrics = metrics_result.scalars().all()
    metric_vals: dict[str, float] = {}
    sample_size = 0
    source = "backtest"
    for m in metrics:
        metric_vals[m.metric_name] = m.metric_value
        sample_size = m.sample_size
        source = m.source

    badge = _badge_for_market(market)
    gates = _gates_for_market(
        cal_bins, sample_size, metric_vals.get("brier"),
        has_direct_data=badge.has_direct_data,
    )

    return MarketAccuracyDetailOut(
        market=market,
        reliability=reliability,
        calibration_bins=cal_bins,
        brier=metric_vals.get("brier"),
        rps=metric_vals.get("rps"),
        log_loss=metric_vals.get("log_loss"),
        hit_rate=metric_vals.get("hit_rate"),
        sample_size=sample_size,
        badge=badge,
        gates=gates,
        source=source,
    )
