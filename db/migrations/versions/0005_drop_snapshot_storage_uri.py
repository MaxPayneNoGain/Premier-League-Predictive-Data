"""drop snapshot storage uri

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-12 14:10:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("snapshots", "storage_uri")


def downgrade() -> None:
    op.add_column("snapshots", sa.Column("storage_uri", sa.Text(), nullable=False))
