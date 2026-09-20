"""Initial schema with all 15 tables.

Revision ID: 001
Revises:
Create Date: 2026-09-20
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "leagues",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("tier", sa.Integer, server_default="1"),
        sa.Column("is_active", sa.Boolean, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("league_id", sa.Integer, sa.ForeignKey("leagues.id"), nullable=False),
        sa.Column("label", sa.String(20), nullable=False),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("league_id", "label", name="uq_season_league_label"),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "team_aliases",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("raw_name", sa.String(200), nullable=False),
        sa.UniqueConstraint("source", "raw_name", name="uq_alias_source_name"),
    )

    op.create_table(
        "referees",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False, unique=True),
        sa.Column("country", sa.String(100)),
    )

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("league_id", sa.Integer, sa.ForeignKey("leagues.id"), nullable=False),
        sa.Column("season_id", sa.Integer, sa.ForeignKey("seasons.id")),
        sa.Column("home_team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("referee_id", sa.Integer, sa.ForeignKey("referees.id")),
        sa.Column("kickoff_utc", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_goals", sa.Integer),
        sa.Column("away_goals", sa.Integer),
        sa.Column("status", sa.String(20), server_default="scheduled"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint(
            "league_id",
            "kickoff_utc",
            "home_team_id",
            "away_team_id",
            name="uq_match_natural_key",
        ),
    )

    op.create_table(
        "elo_ratings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("rating", sa.Float, nullable=False),
        sa.Column("rating_delta", sa.Float, nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("parameters", sa.Text),
        sa.Column("training_window_start", sa.DateTime(timezone=True)),
        sa.Column("training_window_end", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "team_strengths",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_run_id", sa.Integer, sa.ForeignKey("model_runs.id"), nullable=False),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("attack", sa.Float, nullable=False),
        sa.Column("defence", sa.Float, nullable=False),
        sa.Column("home_advantage", sa.Float),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_run_id", sa.Integer, sa.ForeignKey("model_runs.id"), nullable=False),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("home_win_prob", sa.Float, nullable=False),
        sa.Column("draw_prob", sa.Float, nullable=False),
        sa.Column("away_win_prob", sa.Float, nullable=False),
        sa.Column("home_expected_goals", sa.Float),
        sa.Column("away_expected_goals", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "market_predictions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("prediction_id", sa.Integer, sa.ForeignKey("predictions.id"), nullable=False),
        sa.Column("market", sa.String(50), nullable=False),
        sa.Column("selection", sa.String(50), nullable=False),
        sa.Column("probability", sa.Float, nullable=False),
        sa.Column("line", sa.Float),
    )

    op.create_table(
        "calibration_maps",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_run_id", sa.Integer, sa.ForeignKey("model_runs.id"), nullable=False),
        sa.Column("market", sa.String(50), nullable=False),
        sa.Column("bin_lower", sa.Float, nullable=False),
        sa.Column("bin_upper", sa.Float, nullable=False),
        sa.Column("predicted_frequency", sa.Float, nullable=False),
        sa.Column("observed_frequency", sa.Float, nullable=False),
        sa.Column("sample_size", sa.Integer, nullable=False),
    )

    op.create_table(
        "outcomes",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("match_id", sa.Integer, sa.ForeignKey("matches.id"), nullable=False, unique=True),
        sa.Column("result", sa.String(10), nullable=False),
        sa.Column("home_goals", sa.Integer, nullable=False),
        sa.Column("away_goals", sa.Integer, nullable=False),
        sa.Column("settled_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "accuracy_metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("model_run_id", sa.Integer, sa.ForeignKey("model_runs.id"), nullable=False),
        sa.Column("metric_name", sa.String(50), nullable=False),
        sa.Column("metric_value", sa.Float, nullable=False),
        sa.Column("sample_size", sa.Integer, nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "ingest_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(50), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(20), server_default="running"),
        sa.Column("records_processed", sa.Integer, server_default="0"),
        sa.Column("error_message", sa.Text),
    )


def downgrade() -> None:
    op.drop_table("ingest_runs")
    op.drop_table("accuracy_metrics")
    op.drop_table("outcomes")
    op.drop_table("calibration_maps")
    op.drop_table("market_predictions")
    op.drop_table("predictions")
    op.drop_table("team_strengths")
    op.drop_table("model_runs")
    op.drop_table("elo_ratings")
    op.drop_table("matches")
    op.drop_table("referees")
    op.drop_table("team_aliases")
    op.drop_table("teams")
    op.drop_table("seasons")
    op.drop_table("leagues")
