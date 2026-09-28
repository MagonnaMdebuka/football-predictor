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
@click.option("--league", default="E0", help="League code (e.g. E0)")
@click.option("--days-ahead", default=14, type=int, help="Predict fixtures within N days")
@click.option("--dry-run", is_flag=True, help="Log actions without writing to DB")
@click.option("--force", is_flag=True, help="Bypass fingerprint check and re-predict")
def run_cmd(league: str, days_ahead: int, dry_run: bool, force: bool):
    """Fit model and generate predictions for upcoming fixtures."""
    from services.engine.predict.runner import run_predict

    click.echo(f"Running predictions for {league} (next {days_ahead} days)...")
    run_predict(league_code=league, days_ahead=days_ahead, dry_run=dry_run, force=force)
    click.echo("Prediction run complete.")
