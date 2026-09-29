"""Phase 7: calibration maps, accuracy metrics, model-vs-market.

Alter calibration_maps and accuracy_metrics for backtest-sourced data.
Add model_vs_market table for comparing model vs bookmaker probabilities.

Revision ID: 005
Revises: 004
Create Date: 2026-09-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- calibration_maps changes ---
    # Make model_run_id nullable (backtest-sourced maps have no model_run)
    op.alter_column(
        "calibration_maps",
        "model_run_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.add_column(
        "calibration_maps",
        sa.Column("source", sa.String(20), nullable=False, server_default="backtest"),
    )
    op.add_column(
        "calibration_maps",
        sa.Column(
            "league_id", sa.Integer, sa.ForeignKey("leagues.id"), nullable=True
        ),
    )
    op.add_column(
        "calibration_maps",
        sa.Column("selection", sa.String(50), nullable=False, server_default=""),
    )
    op.add_column(
        "calibration_maps",
        sa.Column(
            "is_current", sa.Boolean, nullable=False, server_default=sa.text("true")
        ),
    )
    op.add_column(
        "calibration_maps",
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
    )
    op.add_column(
        "calibration_maps",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_calibration_current",
        "calibration_maps",
        ["market", "selection", "is_current"],
    )

    # --- accuracy_metrics changes ---
    op.alter_column(
        "accuracy_metrics",
        "model_run_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.add_column(
        "accuracy_metrics",
        sa.Column("source", sa.String(20), nullable=False, server_default="backtest"),
    )
    op.add_column(
        "accuracy_metrics",
        sa.Column(
            "league_id", sa.Integer, sa.ForeignKey("leagues.id"), nullable=True
        ),
    )
    op.add_column(
        "accuracy_metrics",
        sa.Column("market", sa.String(50), nullable=True),
    )
    op.create_index(
        "ix_accuracy_metrics_source",
        "accuracy_metrics",
        ["source", "league_id", "market"],
    )

    # --- model_vs_market table ---
    op.create_table(
        "model_vs_market",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "prediction_id",
            sa.Integer,
            sa.ForeignKey("predictions.id"),
            nullable=True,
        ),
        sa.Column("market", sa.String(50), nullable=False),
        sa.Column("selection", sa.String(50), nullable=False),
        sa.Column("model_prob", sa.Float, nullable=False),
        sa.Column("bookmaker_prob", sa.Float, nullable=False),
        sa.Column("bookmaker_source", sa.String(50), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="backtest"),
        sa.Column(
            "league_id", sa.Integer, sa.ForeignKey("leagues.id"), nullable=True
        ),
    )
    op.create_index(
        "ix_model_vs_market_prediction",
        "model_vs_market",
        ["prediction_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_model_vs_market_prediction", table_name="model_vs_market")
    op.drop_table("model_vs_market")

    op.drop_index("ix_accuracy_metrics_source", table_name="accuracy_metrics")
    op.drop_column("accuracy_metrics", "market")
    op.drop_column("accuracy_metrics", "league_id")
    op.drop_column("accuracy_metrics", "source")
    op.alter_column(
        "accuracy_metrics",
        "model_run_id",
        existing_type=sa.Integer(),
        nullable=False,
    )

    op.drop_index("ix_calibration_current", table_name="calibration_maps")
    op.drop_column("calibration_maps", "created_at")
    op.drop_column("calibration_maps", "version")
    op.drop_column("calibration_maps", "is_current")
    op.drop_column("calibration_maps", "selection")
    op.drop_column("calibration_maps", "league_id")
    op.drop_column("calibration_maps", "source")
    op.alter_column(
        "calibration_maps",
        "model_run_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
