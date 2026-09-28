"""Phase 6: predict command schema changes.

Add league_id, fingerprint, n_training_matches, git_commit to model_runs.
Add grid_compressed, n_matches_train to predictions.
Add indexes for current-prediction queries, fixture window, and market joins.

Revision ID: 004
Revises: 003
Create Date: 2026-09-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # model_runs additions
    op.add_column(
        "model_runs",
        sa.Column("league_id", sa.Integer, sa.ForeignKey("leagues.id"), nullable=True),
    )
    op.add_column(
        "model_runs",
        sa.Column("fingerprint", sa.String(64), nullable=True),
    )
    op.create_unique_constraint("uq_model_runs_fingerprint", "model_runs", ["fingerprint"])
    op.add_column(
        "model_runs",
        sa.Column("n_training_matches", sa.Integer, nullable=True),
    )
    op.add_column(
        "model_runs",
        sa.Column("git_commit", sa.String(40), nullable=True),
    )

    # predictions additions
    op.add_column(
        "predictions",
        sa.Column("grid_compressed", sa.LargeBinary, nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("n_matches_train", sa.Integer, nullable=True),
    )

    # Indexes
    op.create_index(
        "ix_predictions_match_created",
        "predictions",
        ["match_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_model_runs_league_created",
        "model_runs",
        ["league_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_market_predictions_prediction",
        "market_predictions",
        ["prediction_id"],
    )
    op.create_index(
        "ix_matches_status_kickoff",
        "matches",
        ["status", "kickoff_utc"],
    )


def downgrade() -> None:
    op.drop_index("ix_matches_status_kickoff", table_name="matches")
    op.drop_index("ix_market_predictions_prediction", table_name="market_predictions")
    op.drop_index("ix_model_runs_league_created", table_name="model_runs")
    op.drop_index("ix_predictions_match_created", table_name="predictions")

    op.drop_column("predictions", "n_matches_train")
    op.drop_column("predictions", "grid_compressed")

    op.drop_constraint("uq_model_runs_fingerprint", "model_runs", type_="unique")
    op.drop_column("model_runs", "git_commit")
    op.drop_column("model_runs", "n_training_matches")
    op.drop_column("model_runs", "fingerprint")
    op.drop_column("model_runs", "league_id")
