"""Phase 1 ingest schema changes.

Add capability flags to leagues, rename team.name to canonical_name,
extend matches with detailed stats, add match_source_rows table,
update ingest_runs with job tracking.

Revision ID: 002
Revises: 001
Create Date: 2026-09-21
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- leagues: add capability flags and source codes ---
    op.add_column("leagues", sa.Column("fd_couk_code", sa.String(10)))
    op.add_column("leagues", sa.Column("fd_org_code", sa.Integer))
    op.add_column("leagues", sa.Column("has_corners", sa.Boolean, server_default="false"))
    op.add_column("leagues", sa.Column("has_cards", sa.Boolean, server_default="false"))
    op.add_column("leagues", sa.Column("has_xg", sa.Boolean, server_default="false"))

    # --- teams: rename name → canonical_name ---
    op.drop_constraint("teams_name_key", "teams", type_="unique")
    op.alter_column("teams", "name", new_column_name="canonical_name")
    op.create_unique_constraint("uq_teams_canonical_name", "teams", ["canonical_name"])

    # --- team_aliases: make team_id nullable, add score + confirmed ---
    op.alter_column("team_aliases", "team_id", nullable=True)
    op.add_column("team_aliases", sa.Column("score", sa.Float))
    op.add_column("team_aliases", sa.Column("confirmed", sa.Boolean, server_default="false"))

    # --- matches: rename goal columns ---
    op.alter_column("matches", "home_goals", new_column_name="ft_home_goals")
    op.alter_column("matches", "away_goals", new_column_name="ft_away_goals")

    # --- matches: add detailed stat columns ---
    op.add_column("matches", sa.Column("ht_home_goals", sa.Integer))
    op.add_column("matches", sa.Column("ht_away_goals", sa.Integer))
    op.add_column("matches", sa.Column("home_shots", sa.Integer))
    op.add_column("matches", sa.Column("away_shots", sa.Integer))
    op.add_column("matches", sa.Column("home_shots_on_target", sa.Integer))
    op.add_column("matches", sa.Column("away_shots_on_target", sa.Integer))
    op.add_column("matches", sa.Column("home_fouls", sa.Integer))
    op.add_column("matches", sa.Column("away_fouls", sa.Integer))
    op.add_column("matches", sa.Column("home_corners", sa.Integer))
    op.add_column("matches", sa.Column("away_corners", sa.Integer))
    op.add_column("matches", sa.Column("home_yellows", sa.Integer))
    op.add_column("matches", sa.Column("away_yellows", sa.Integer))
    op.add_column("matches", sa.Column("home_reds", sa.Integer))
    op.add_column("matches", sa.Column("away_reds", sa.Integer))
    op.add_column("matches", sa.Column("kickoff_time_known", sa.Boolean, server_default="true"))
    op.add_column("matches", sa.Column("no_crowd", sa.Boolean, server_default="false"))
    op.add_column("matches", sa.Column("source_match_id", sa.String(50)))

    # --- matches: change unique constraint ---
    # Make season_id NOT NULL (backfill NULLs first if any exist)
    op.execute("UPDATE matches SET season_id = 0 WHERE season_id IS NULL")
    op.alter_column("matches", "season_id", nullable=False)

    # Drop old unique key (league, kickoff_utc, home, away)
    op.drop_constraint("uq_match_natural_key", "matches", type_="unique")
    # New natural key: league + season + home + away
    op.create_unique_constraint(
        "uq_match_natural_key",
        "matches",
        ["league_id", "season_id", "home_team_id", "away_team_id"],
    )

    # --- ingest_runs: rename records_processed → rows_written, add job + rows_skipped ---
    op.alter_column("ingest_runs", "records_processed", new_column_name="rows_written")
    op.add_column("ingest_runs", sa.Column("job", sa.String(50)))
    op.add_column("ingest_runs", sa.Column("rows_skipped", sa.Integer, server_default="0"))

    # --- match_source_rows: new table ---
    op.create_table(
        "match_source_rows",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("raw", sa.JSON, nullable=False),
        sa.Column("file_checksum", sa.String(64)),
        sa.UniqueConstraint("match_id", "source", name="uq_match_source_row"),
    )


def downgrade() -> None:
    # --- match_source_rows ---
    op.drop_table("match_source_rows")

    # --- ingest_runs ---
    op.drop_column("ingest_runs", "rows_skipped")
    op.drop_column("ingest_runs", "job")
    op.alter_column("ingest_runs", "rows_written", new_column_name="records_processed")

    # --- matches: restore unique constraint ---
    op.drop_constraint("uq_match_natural_key", "matches", type_="unique")
    op.alter_column("matches", "season_id", nullable=True)
    op.create_unique_constraint(
        "uq_match_natural_key",
        "matches",
        ["league_id", "kickoff_utc", "home_team_id", "away_team_id"],
    )

    # --- matches: drop stat columns ---
    op.drop_column("matches", "source_match_id")
    op.drop_column("matches", "no_crowd")
    op.drop_column("matches", "kickoff_time_known")
    op.drop_column("matches", "home_reds")
    op.drop_column("matches", "away_reds")
    op.drop_column("matches", "home_yellows")
    op.drop_column("matches", "away_yellows")
    op.drop_column("matches", "home_corners")
    op.drop_column("matches", "away_corners")
    op.drop_column("matches", "home_fouls")
    op.drop_column("matches", "away_fouls")
    op.drop_column("matches", "home_shots_on_target")
    op.drop_column("matches", "away_shots_on_target")
    op.drop_column("matches", "home_shots")
    op.drop_column("matches", "away_shots")
    op.drop_column("matches", "ht_home_goals")
    op.drop_column("matches", "ht_away_goals")

    # --- matches: rename goal columns back ---
    op.alter_column("matches", "ft_away_goals", new_column_name="away_goals")
    op.alter_column("matches", "ft_home_goals", new_column_name="home_goals")

    # --- team_aliases ---
    op.drop_column("team_aliases", "confirmed")
    op.drop_column("team_aliases", "score")
    op.alter_column("team_aliases", "team_id", nullable=False)

    # --- teams ---
    op.drop_constraint("uq_teams_canonical_name", "teams", type_="unique")
    op.alter_column("teams", "canonical_name", new_column_name="name")
    op.create_unique_constraint("teams_name_key", "teams", ["name"])

    # --- leagues ---
    op.drop_column("leagues", "has_xg")
    op.drop_column("leagues", "has_cards")
    op.drop_column("leagues", "has_corners")
    op.drop_column("leagues", "fd_org_code")
    op.drop_column("leagues", "fd_couk_code")
