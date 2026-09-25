"""JSON report serialisation and deserialisation.

Reports are written with deterministic formatting for byte-identical
reproducibility: sort_keys=True, indent=2, floats rounded to 10 decimal
places, Unix line endings.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from services.engine.backtest.types import (
    BacktestConfig,
    BacktestReport,
    BookmakerOddsCols,
    CountCalibrationSummary,
    CountMetricSummary,
    GateDetail,
    GoalCalibration,
    MatchPrediction,
    MetricSet,
    SeasonMetrics,
)

FLOAT_PRECISION = 10
SCHEMA_VERSION = "2.0.0"


class _DeterministicEncoder(json.JSONEncoder):
    """JSON encoder that rounds floats and converts numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), FLOAT_PRECISION)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _round_floats(obj):
    """Recursively round all floats to FLOAT_PRECISION decimal places."""
    if isinstance(obj, float):
        return round(obj, FLOAT_PRECISION)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(x) for x in obj]
    return obj


def report_to_dict(report: BacktestReport) -> dict:
    """Convert a BacktestReport to a JSON-serialisable dict."""
    d = asdict(report)
    return _round_floats(d)


def serialise_report(report: BacktestReport) -> str:
    """Serialise a BacktestReport to a deterministic JSON string.

    Uses sort_keys, 2-space indent, 10dp float precision, and Unix line endings.
    """
    d = report_to_dict(report)
    result = json.dumps(d, cls=_DeterministicEncoder, sort_keys=True, indent=2)
    # Ensure Unix line endings
    result = result.replace("\r\n", "\n")
    # Trailing newline
    if not result.endswith("\n"):
        result += "\n"
    return result


def deserialise_report(json_str: str) -> BacktestReport:
    """Deserialise a JSON string back to a BacktestReport."""
    d = json.loads(json_str)

    cfg_d = d["config"]
    config = BacktestConfig(
        held_out_seasons=tuple(cfg_d["held_out_seasons"]),
        training_start_season=cfg_d["training_start_season"],
        xi=cfg_d["xi"],
        refit_step=cfg_d["refit_step"],
        weekly_refit_day=cfg_d["weekly_refit_day"],
        rho_bounds=tuple(cfg_d["rho_bounds"]),
        max_goals=cfg_d["max_goals"],
        seed=cfg_d["seed"],
        bookmaker_odds_cols=BookmakerOddsCols(**cfg_d["bookmaker_odds_cols"]),
        league_code=cfg_d.get("league_code"),
        corners_xi=cfg_d.get("corners_xi"),
        cards_xi=cfg_d.get("cards_xi"),
        min_referee_matches=cfg_d.get("min_referee_matches", 20),
    )

    def _to_metric_set(md: dict) -> MetricSet:
        return MetricSet(**md)

    def _to_season_metrics(sd: dict) -> SeasonMetrics:
        ablation_data = sd.get("ablation")
        return SeasonMetrics(
            season=sd["season"],
            model=_to_metric_set(sd["model"]),
            uniform=_to_metric_set(sd["uniform"]),
            base_rate=_to_metric_set(sd["base_rate"]),
            independent_poisson=_to_metric_set(sd["independent_poisson"]),
            ablation=_to_metric_set(ablation_data) if ablation_data else None,
            bookmaker=_to_metric_set(sd["bookmaker"]) if sd["bookmaker"] else None,
            bookmaker_exclusion_count=sd["bookmaker_exclusion_count"],
            early_season=_to_metric_set(sd["early_season"]) if sd["early_season"] else None,
        )

    seasons = [_to_season_metrics(s) for s in d["seasons"]]
    combined = _to_season_metrics(d["combined"])
    early_combined = (
        _to_season_metrics(d["early_season_combined"])
        if d["early_season_combined"]
        else None
    )

    pred_defaults = {
        "fallback_teams": [],
        "ablation_home": 0.0,
        "ablation_draw": 0.0,
        "ablation_away": 0.0,
        "lambda_home": 0.0,
        "lambda_away": 0.0,
    }
    predictions = [
        MatchPrediction(**{**pred_defaults, **p})
        for p in d["predictions"]
    ]
    gate_details = [GateDetail(**g) for g in d["gate_details"]]

    goal_cal_data = d.get("goal_calibration")
    goal_cal = GoalCalibration(**goal_cal_data) if goal_cal_data else None

    # Count model fields (backward compatible — absent in v1.0.0 reports)
    corner_preds = d.get("corner_predictions", [])
    card_preds = d.get("card_predictions", [])

    def _to_count_metrics(data: dict | None) -> CountMetricSummary | None:
        if data is None:
            return None
        # Convert string keys back to float for per_line_brier
        plb = {float(k): v for k, v in data["per_line_brier"].items()}
        return CountMetricSummary(
            mean_brier=data["mean_brier"],
            per_line_brier=plb,
            n_predictions=data["n_predictions"],
            mean_predicted_total=data["mean_predicted_total"],
            mean_actual_total=data["mean_actual_total"],
        )

    def _to_count_cal(data: dict | None) -> CountCalibrationSummary | None:
        if data is None:
            return None
        return CountCalibrationSummary(**data)

    corner_metrics = _to_count_metrics(d.get("corner_metrics"))
    card_metrics = _to_count_metrics(d.get("card_metrics"))
    corner_cal = _to_count_cal(d.get("corner_calibration"))
    card_cal = _to_count_cal(d.get("card_calibration"))
    corner_gate_details = [GateDetail(**g) for g in d.get("corner_gate_details", [])]
    card_gate_details = [GateDetail(**g) for g in d.get("card_gate_details", [])]

    return BacktestReport(
        schema_version=d["schema_version"],
        created_at=d["created_at"],
        git_commit=d["git_commit"],
        config=config,
        seasons=seasons,
        combined=combined,
        early_season_combined=early_combined,
        predictions=predictions,
        gate_passed=d["gate_passed"],
        gate_details=gate_details,
        baseline_configs=d.get("baseline_configs", {}),
        goal_calibration=goal_cal,
        corner_predictions=corner_preds,
        card_predictions=card_preds,
        corner_metrics=corner_metrics,
        card_metrics=card_metrics,
        corner_calibration=corner_cal,
        card_calibration=card_cal,
        corner_gate_passed=d.get("corner_gate_passed"),
        card_gate_passed=d.get("card_gate_passed"),
        corner_gate_details=corner_gate_details,
        card_gate_details=card_gate_details,
    )


def generate_filename(git_commit: str | None = None) -> str:
    """Generate a report filename: <ISO-timestamp>_<short-git-hash>.json.

    Args:
        git_commit: short git hash (auto-detected if None)

    Returns:
        Filename string.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if git_commit is None:
        git_commit = get_git_commit()
    return f"{timestamp}_{git_commit}.json"


def get_git_commit() -> str:
    """Get the short git commit hash of the current HEAD."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_report(report: BacktestReport, output_dir: str | Path) -> Path:
    """Write a backtest report to a JSON file.

    Args:
        report: the report to write
        output_dir: directory to write the file to

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = generate_filename(report.git_commit)
    filepath = output_dir / filename
    content = serialise_report(report)
    filepath.write_text(content, encoding="utf-8", newline="")
    return filepath


def read_report(filepath: str | Path) -> BacktestReport:
    """Read a backtest report from a JSON file."""
    content = Path(filepath).read_text(encoding="utf-8")
    return deserialise_report(content)


def reports_identical(path_a: str | Path, path_b: str | Path) -> bool:
    """Check whether two report files are byte-identical."""
    content_a = Path(path_a).read_bytes()
    content_b = Path(path_b).read_bytes()
    return content_a == content_b
