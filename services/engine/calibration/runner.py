"""Calibration bootstrap runner — DB orchestration.

Reads a backtest JSON, builds calibration maps, computes accuracy metrics,
and writes results to the database.
"""

from __future__ import annotations

import glob
import logging
import os

from sqlalchemy import select, update

from db.models import AccuracyMetric, CalibrationMap, League, ModelVsMarket
from services.engine.calibration.accuracy import compute_accuracy_metrics
from services.engine.calibration.bootstrap import (
    build_calibration_maps_from_backtest,
    derive_market_outcomes,
    load_backtest_predictions,
)
from services.engine.calibration.reliability import compute_reliability_data

logger = logging.getLogger(__name__)


def _find_latest_backtest() -> str:
    """Find the most recent backtest JSON in the backtests/ directory."""
    pattern = os.path.join("backtests", "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No backtest JSON files found in backtests/")
    return files[-1]


def _parse_market_selection(market_key: str) -> tuple[str, str]:
    """Split a market_selection key like 'match_result_home' into (market, selection).

    Convention: last segment after the last underscore is the selection,
    except for compound names like 'over_under_2.5_over'.
    """
    known_markets = {
        "match_result": "match_result",
        "over_under_2.5": "over_under_2.5",
        "over_under_1.5": "over_under_1.5",
        "over_under_3.5": "over_under_3.5",
        "btts": "btts",
    }

    for prefix, market in sorted(known_markets.items(), key=lambda x: -len(x[0])):
        if market_key.startswith(prefix + "_"):
            selection = market_key[len(prefix) + 1:]
            return market, selection

    return market_key, ""


def run_bootstrap(
    backtest_path: str | None = None,
    league_code: str = "E0",
) -> None:
    """Bootstrap calibration maps and accuracy metrics from a backtest report."""
    from services.engine.ingest.db_session import get_session

    if backtest_path is None:
        backtest_path = _find_latest_backtest()

    logger.info("Loading backtest from %s", backtest_path)

    # Build calibration maps (pure library)
    calibration_maps = build_calibration_maps_from_backtest(backtest_path)
    predictions = load_backtest_predictions(backtest_path)
    market_outcomes = derive_market_outcomes(predictions)

    with get_session() as session:
        # Look up league
        league = session.execute(
            select(League).where(League.fd_couk_code == league_code)
        ).scalar_one_or_none()
        if league is None:
            logger.error("League %s not found in database", league_code)
            return

        league_id = league.id

        # Mark existing calibration maps as not current
        session.execute(
            update(CalibrationMap)
            .where(CalibrationMap.league_id == league_id)
            .values(is_current=False)
        )

        # Compute next version
        from sqlalchemy import func

        max_version = session.execute(
            select(func.coalesce(func.max(CalibrationMap.version), 0))
            .where(CalibrationMap.league_id == league_id)
        ).scalar()
        next_version = max_version + 1

        # Write calibration maps
        cal_count = 0
        for market_key, bins in calibration_maps.items():
            market, selection = _parse_market_selection(market_key)
            for b in bins:
                session.add(CalibrationMap(
                    market=market,
                    selection=selection,
                    bin_lower=b["bin_lower"],
                    bin_upper=b["bin_upper"],
                    predicted_frequency=b["predicted_frequency"],
                    observed_frequency=b["observed_frequency"],
                    sample_size=b["sample_size"],
                    source="backtest",
                    league_id=league_id,
                    is_current=True,
                    version=next_version,
                ))
                cal_count += 1

        logger.info("Wrote %d calibration map rows (version %d)", cal_count, next_version)

        # Compute and write accuracy metrics per market
        metric_count = 0
        for market_key, obs_list in market_outcomes.items():
            if not obs_list:
                continue
            market, selection = _parse_market_selection(market_key)
            metrics = compute_accuracy_metrics(obs_list, market_key)
            n = metrics["sample_size"]
            if n == 0:
                continue

            for metric_name in ["brier", "log_loss", "hit_rate", "rps"]:
                value = metrics[metric_name]
                if value is not None:
                    session.add(AccuracyMetric(
                        metric_name=metric_name,
                        metric_value=value,
                        sample_size=n,
                        source="backtest",
                        league_id=league_id,
                        market=market_key,
                    ))
                    metric_count += 1

        logger.info("Wrote %d accuracy metric rows", metric_count)

        # Write model-vs-market comparison (1X2 only — backtest has bookmaker probs)
        mvm_count = 0
        for p in predictions:
            bk_home = p.get("bookmaker_home")
            if bk_home is None:
                continue
            for sel, model_key, bk_key in [
                ("home", "model_home", "bookmaker_home"),
                ("draw", "model_draw", "bookmaker_draw"),
                ("away", "model_away", "bookmaker_away"),
            ]:
                session.add(ModelVsMarket(
                    market="match_result",
                    selection=sel,
                    model_prob=p[model_key],
                    bookmaker_prob=p[bk_key],
                    bookmaker_source="average",
                    source="backtest",
                    league_id=league_id,
                ))
                mvm_count += 1

        logger.info("Wrote %d model-vs-market rows", mvm_count)

        # Compute reliability data per market and log summary
        for market_key, obs_list in market_outcomes.items():
            if market_key == "match_result" or not obs_list:
                continue
            import numpy as np

            predicted = np.array([o["predicted_prob"] for o in obs_list])
            observed = np.array([o["observed"] for o in obs_list])
            rel = compute_reliability_data(predicted, observed)
            logger.info(
                "  %s: MCE=%.3f, max_error=%.3f, n=%d",
                market_key,
                rel["mean_calibration_error"],
                rel["calibration_error"],
                len(obs_list),
            )

        session.commit()
        logger.info("Bootstrap complete for league %s", league_code)
