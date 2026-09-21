"""Click CLI commands for football data ingestion."""

import logging
import sys

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)


@click.group()
def ingest():
    """Football data ingestion commands."""
    pass


@ingest.command("seed")
def seed_cmd():
    """Seed leagues, seasons, and team aliases."""
    from services.engine.ingest.runner import run_seed

    click.echo("Seeding leagues, seasons, and team aliases...")
    run_seed()
    click.echo("Seed complete.")


@ingest.command("csv-backfill")
def csv_backfill_cmd():
    """Backfill season CSVs from football-data.co.uk."""
    from services.engine.ingest.runner import run_csv_backfill

    click.echo("Starting CSV backfill...")
    run_csv_backfill()
    click.echo("CSV backfill complete.")


@ingest.command("fixtures-sync")
def fixtures_sync_cmd():
    """Sync upcoming fixtures from football-data.org."""
    from services.engine.ingest.runner import run_fixtures_sync

    click.echo("Syncing fixtures from football-data.org...")
    run_fixtures_sync()
    click.echo("Fixture sync complete.")


@ingest.command("verify-ingest")
def verify_ingest_cmd():
    """Run quality checks on ingested data."""
    from services.engine.ingest.runner import run_verify_ingest

    passed = run_verify_ingest()
    sys.exit(0 if passed else 1)


@ingest.group("aliases")
def aliases_group():
    """Manage team alias resolution."""
    pass


@aliases_group.command("review")
def aliases_review_cmd():
    """Show unconfirmed team aliases."""
    from services.engine.ingest.runner import run_aliases_review

    run_aliases_review()


@aliases_group.command("confirm")
@click.option("--source", required=True, help="Source name (e.g., fd_couk, fd_org)")
@click.option("--raw-name", required=True, help="Raw team name from source")
@click.option("--team", required=True, help="Canonical team name to map to")
def aliases_confirm_cmd(source: str, raw_name: str, team: str):
    """Manually confirm a team alias mapping."""
    from services.engine.ingest.runner import run_aliases_confirm

    run_aliases_confirm(source, raw_name, team)


if __name__ == "__main__":
    ingest()
