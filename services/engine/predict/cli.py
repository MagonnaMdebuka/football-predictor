"""Click CLI commands for the predict pipeline."""

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)


@click.group()
def predict():
    """Prediction generation commands."""
    pass


@predict.command("run")
@click.option("--league", default=None, help="League code (e.g. E0)")
@click.option("--all-active", is_flag=True, help="Run for all leagues where is_active=true")
@click.option("--days-ahead", default=14, type=int, help="Predict fixtures within N days")
@click.option("--dry-run", is_flag=True, help="Log actions without writing to DB")
@click.option("--force", is_flag=True, help="Bypass fingerprint check and re-predict")
def run_cmd(league: str | None, all_active: bool, days_ahead: int, dry_run: bool, force: bool):
    """Fit model and generate predictions for upcoming fixtures."""
    from services.engine.predict.runner import run_predict

    leagues = _resolve_leagues(league, all_active)
    for code in leagues:
        click.echo(f"Running predictions for {code} (next {days_ahead} days)...")
        run_predict(league_code=code, days_ahead=days_ahead, dry_run=dry_run, force=force)
        click.echo(f"Prediction run complete for {code}.")


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
