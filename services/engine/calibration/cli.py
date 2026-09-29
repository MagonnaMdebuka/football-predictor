"""Click CLI commands for calibration."""

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)


@click.group()
def calibration():
    """Calibration and accuracy commands."""
    pass


@calibration.command("bootstrap")
@click.option(
    "--backtest-path",
    default=None,
    help="Path to backtest JSON (default: latest in backtests/)",
)
@click.option("--league", default="E0", help="League code (e.g. E0)")
def bootstrap_cmd(backtest_path: str | None, league: str):
    """Bootstrap calibration maps from a backtest report."""
    from services.engine.calibration.runner import run_bootstrap

    click.echo(f"Bootstrapping calibration for {league}...")
    run_bootstrap(backtest_path=backtest_path, league_code=league)
    click.echo("Bootstrap complete.")


@calibration.command("refresh")
@click.option("--league", default="E0", help="League code (e.g. E0)")
@click.option("--window", default=2000, type=int, help="Rolling window size")
def refresh_cmd(league: str, window: int):
    """Refresh calibration from live predictions (placeholder)."""
    click.echo(f"Live calibration refresh for {league} (window={window}) — not yet implemented.")
