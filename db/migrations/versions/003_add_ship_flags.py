"""Add ship_corners and ship_cards flags to leagues.

These flags control whether corner and card markets are published
to the UI. Separate from has_corners/has_cards which control whether
the backtest runs count models.

Revision ID: 003
Revises: 002
Create Date: 2026-09-25
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("leagues", sa.Column("ship_corners", sa.Boolean, server_default="false"))
    op.add_column("leagues", sa.Column("ship_cards", sa.Boolean, server_default="false"))


def downgrade() -> None:
    op.drop_column("leagues", "ship_cards")
    op.drop_column("leagues", "ship_corners")
