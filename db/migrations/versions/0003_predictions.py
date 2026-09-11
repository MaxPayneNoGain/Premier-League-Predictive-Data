"""predictions

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-10 18:05:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("match_id", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=32), nullable=False),
        sa.Column("predicted_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("trained_through", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("home_win", sa.Float(), nullable=False),
        sa.Column("draw", sa.Float(), nullable=False),
        sa.Column("away_win", sa.Float(), nullable=False),
        sa.Column("home_xg", sa.Float(), nullable=False),
        sa.Column("away_xg", sa.Float(), nullable=False),
        sa.Column("rho", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("match_id", "model", "predicted_at", name="uq_predictions_run"),
    )
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])


def downgrade() -> None:
    op.drop_index("ix_predictions_match_id", table_name="predictions")
    op.drop_table("predictions")
