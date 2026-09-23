"""Click CLI for backtest commands: run, compare, summary."""

from __future__ import annotations

import click
import pandas as pd

from services.engine.backtest.report import (
    read_report,
    reports_identical,
    write_report,
)
from services.engine.backtest.types import BacktestConfig


@click.group()
def backtest():
    """Walk-forward backtest harness for Dixon-Coles model."""


@backtest.command()
@click.option("--xi", default=0.0065, type=float, help="Time-decay rate parameter.")
@click.option(
    "--refit-step",
    default="per_date",
    type=click.Choice(["per_date", "weekly"]),
    help="Refit schedule: per_date (accurate) or weekly (fast).",
)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
@click.option(
    "--from-csv",
    default=None,
    type=click.Path(exists=True),
    help="Path to CSV data file (default: uses data/raw/).",
)
@click.option(
    "--output-dir",
    default="backtests",
    type=click.Path(),
    help="Directory for output reports.",
)
def run(xi: float, refit_step: str, seed: int, from_csv: str | None, output_dir: str):
    """Run a walk-forward backtest."""
    config = BacktestConfig(
        held_out_seasons=("2024-25", "2025-26"),
        training_start_season="2019-20",
        xi=xi,
        refit_step=refit_step,
        seed=seed,
    )

    if from_csv:
        df = _load_csv(from_csv)
    else:
        click.echo("Error: --from-csv is required (DB loading not yet implemented).", err=True)
        raise SystemExit(1)

    click.echo(f"Running backtest (refit_step={refit_step}, xi={xi}, seed={seed})...")

    from services.engine.backtest.harness import run_backtest

    report = run_backtest(df, config)

    filepath = write_report(report, output_dir)
    click.echo(f"Report written to: {filepath}")

    _print_gate_summary(report)


@backtest.command()
@click.argument("report_a", type=click.Path(exists=True))
@click.argument("report_b", type=click.Path(exists=True))
def compare(report_a: str, report_b: str):
    """Compare two reports for byte-identical output."""
    if reports_identical(report_a, report_b):
        click.echo("PASS — reports are byte-identical.")
    else:
        click.echo("FAIL — reports differ.")
        raise SystemExit(1)


@backtest.command()
@click.argument("report_path", type=click.Path(exists=True))
def summary(report_path: str):
    """Print summary of a backtest JSON report."""
    report = read_report(report_path)

    click.echo(f"Schema version: {report.schema_version}")
    click.echo(f"Created: {report.created_at}")
    click.echo(f"Git commit: {report.git_commit}")
    click.echo(f"Predictions: {len(report.predictions)}")
    click.echo(f"Held-out seasons: {', '.join(report.config.held_out_seasons)}")
    click.echo(f"Refit step: {report.config.refit_step}")
    click.echo(f"Xi: {report.config.xi}")
    click.echo()

    for sm in report.seasons:
        click.echo(f"--- {sm.season} ---")
        click.echo(f"  Model RPS:  {sm.model.rps:.6f}  Log Loss: {sm.model.log_loss:.6f}  "
                    f"Hit Rate: {sm.model.hit_rate:.3f}  (n={sm.model.n_matches})")
        click.echo(f"  Base Rate:  {sm.base_rate.rps:.6f}  Log Loss: {sm.base_rate.log_loss:.6f}")
        click.echo(f"  Indep Poi:  {sm.independent_poisson.rps:.6f}  "
                    f"Log Loss: {sm.independent_poisson.log_loss:.6f}")
        if sm.bookmaker:
            click.echo(f"  Bookmaker:  {sm.bookmaker.rps:.6f}  "
                        f"Log Loss: {sm.bookmaker.log_loss:.6f}  "
                        f"(excluded: {sm.bookmaker_exclusion_count})")
        click.echo()

    click.echo("=== Combined ===")
    c = report.combined
    click.echo(f"  Model RPS:  {c.model.rps:.6f}  Log Loss: {c.model.log_loss:.6f}  "
                f"Hit Rate: {c.model.hit_rate:.3f}  (n={c.model.n_matches})")
    click.echo()

    _print_gate_summary(report)


def _print_gate_summary(report):
    """Print gate check results."""
    click.echo("=== Gate ===")
    for detail in report.gate_details:
        status = "PASS" if detail.passed else "FAIL"
        click.echo(f"  [{status}] {detail.name}: {detail.message}")
    click.echo()
    if report.gate_passed:
        click.echo("Gate: PASS")
    else:
        click.echo("Gate: FAIL")


def _load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and normalise columns for the backtest harness."""
    raw = pd.read_csv(path)

    # Determine column mapping based on available columns
    if "Date" in raw.columns:
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], dayfirst=True),
            "home_team": raw.get("HomeTeam", raw.get("HT")),
            "away_team": raw.get("AwayTeam", raw.get("AT")),
            "home_goals": raw.get("FTHG", raw.get("HG")),
            "away_goals": raw.get("FTAG", raw.get("AG")),
            "ftr": raw.get("FTR", raw.get("Res")),
        })
    elif "date" in raw.columns:
        df = raw[["date", "home_team", "away_team", "home_goals", "away_goals", "ftr"]].copy()
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise click.ClickException(
            "CSV must have either 'Date' or 'date' column."
        )

    # Infer season from date if not present
    if "season" not in raw.columns and "Season" not in raw.columns:
        df["season"] = df["date"].apply(_infer_season)
    else:
        df["season"] = raw.get("season", raw.get("Season"))

    # Carry through source row as JSON for bookmaker odds
    odds_cols = ["PSCH", "PSCD", "PSCA", "AvgCH", "AvgCD", "AvgCA"]
    available_odds = [c for c in odds_cols if c in raw.columns]
    if available_odds:
        import json
        df["source_row_raw"] = raw[available_odds].apply(
            lambda row: json.dumps({k: v for k, v in row.items() if pd.notna(v)}),
            axis=1,
        )

    return df


def _infer_season(dt) -> str:
    """Infer season string from a date (e.g. Aug 2024 → '2024-25')."""
    if dt.month >= 7:
        return f"{dt.year}-{str(dt.year + 1)[-2:]}"
    return f"{dt.year - 1}-{str(dt.year)[-2:]}"
