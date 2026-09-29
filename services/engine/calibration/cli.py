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
@click.option("--league", default=None, help="League code (e.g. E0)")
@click.option("--all-active", is_flag=True, help="Run for all leagues where is_active=true")
def bootstrap_cmd(backtest_path: str | None, league: str | None, all_active: bool):
    """Bootstrap calibration maps from a backtest report."""
    from services.engine.calibration.runner import run_bootstrap

    leagues = _resolve_leagues(league, all_active)
    for code in leagues:
        click.echo(f"Bootstrapping calibration for {code}...")
        run_bootstrap(backtest_path=backtest_path, league_code=code)
        click.echo(f"Bootstrap complete for {code}.")


@calibration.command("refresh")
@click.option("--league", default=None, help="League code (e.g. E0)")
@click.option("--all-active", is_flag=True, help="Run for all leagues where is_active=true")
@click.option("--window", default=2000, type=int, help="Rolling window size")
def refresh_cmd(league: str | None, all_active: bool, window: int):
    """Refresh calibration from live predictions (placeholder)."""
    leagues = _resolve_leagues(league, all_active)
    for code in leagues:
        click.echo(
            f"Live calibration refresh for {code} (window={window}) — not yet implemented."
        )


def _resolve_leagues(league: str | None, all_active: bool) -> list[str]:
    """Return the list of league codes to process."""
    if all_active and league:
        raise click.UsageError("Use --league or --all-active, not both.")
    if all_active:
        from services.engine.cli_utils import get_active_league_codes

        codes = get_active_league_codes()
        if not codes:
            raise click.ClickException("No active leagues found in the database.")
        click.echo(f"Active leagues: {', '.join(codes)}")
        return codes
    return [league or "E0"]
