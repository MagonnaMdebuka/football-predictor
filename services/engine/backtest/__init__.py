"""Walk-forward backtest harness for Dixon-Coles model.

Public API:
    run_backtest       — run a walk-forward backtest, returns BacktestReport
    BacktestConfig     — configuration dataclass
    BacktestReport     — full report with predictions and metrics
    MetricSet          — evaluation metrics (RPS, log loss, Brier, hit rate)
    MatchPrediction    — per-match prediction record
    SeasonMetrics      — per-season metric breakdown
    BookmakerOddsCols  — bookmaker odds column configuration
    serialise_report   — deterministic JSON serialisation
    deserialise_report — JSON deserialisation
    write_report       — write report to file
    read_report        — read report from file
    reports_identical  — byte-identical comparison
"""

from services.engine.backtest.harness import run_backtest
from services.engine.backtest.report import (
    deserialise_report,
    read_report,
    reports_identical,
    serialise_report,
    write_report,
)
from services.engine.backtest.types import (
    BacktestConfig,
    BacktestReport,
    BookmakerOddsCols,
    MatchPrediction,
    MetricSet,
    SeasonMetrics,
)

__all__ = [
    "run_backtest",
    "BacktestConfig",
    "BacktestReport",
    "BookmakerOddsCols",
    "MatchPrediction",
    "MetricSet",
    "SeasonMetrics",
    "serialise_report",
    "deserialise_report",
    "write_report",
    "read_report",
    "reports_identical",
]
