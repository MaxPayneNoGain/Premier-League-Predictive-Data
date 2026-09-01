"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-31 20:38:54.260166
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fixtures",
        sa.Column("match_id", sa.String(length=128), nullable=False),
        sa.Column("season", sa.String(length=9), nullable=False),
        sa.Column("gameweek", sa.Integer(), nullable=False),
        sa.Column("tournament", sa.String(length=32), nullable=False),
        sa.Column("kickoff_time", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("home_team_id", sa.Integer(), nullable=True),
        sa.Column("away_team_id", sa.Integer(), nullable=True),
        sa.Column("home_team_elo", sa.Float(), nullable=True),
        sa.Column("away_team_elo", sa.Float(), nullable=True),
        sa.Column("home_score", sa.Integer(), nullable=True),
        sa.Column("away_score", sa.Integer(), nullable=True),
        sa.Column("finished", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("match_id"),
    )
    op.create_index("ix_fixtures_season_gameweek", "fixtures", ["season", "gameweek"], unique=False)
    op.create_table(
        "players",
        sa.Column("season", sa.String(length=9), nullable=False),
        sa.Column("fpl_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.Integer(), nullable=False),
        sa.Column("first_name", sa.String(length=64), nullable=False),
        sa.Column("second_name", sa.String(length=64), nullable=False),
        sa.Column("web_name", sa.String(length=64), nullable=False),
        sa.Column("team_code", sa.Integer(), nullable=False),
        sa.Column("position", sa.String(length=16), nullable=False),
        sa.PrimaryKeyConstraint("season", "fpl_id"),
    )
    op.create_index("ix_players_code", "players", ["code"], unique=False)
    op.create_table(
        "snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("season", sa.String(length=9), nullable=False),
        sa.Column("fetched_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("source_ref", sa.String(length=64), nullable=False),
        sa.Column("storage_uri", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source", "season", "content_hash", name="uq_snapshots_content"),
    )
    op.create_index(
        "ix_snapshots_source_fetched_at", "snapshots", ["source", "fetched_at"], unique=False
    )
    op.create_table(
        "teams",
        sa.Column("season", sa.String(length=9), nullable=False),
        sa.Column("fpl_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("short_name", sa.String(length=8), nullable=False),
        sa.PrimaryKeyConstraint("season", "fpl_id"),
    )
    op.create_index("ix_teams_code", "teams", ["code"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_teams_code", table_name="teams")
    op.drop_table("teams")
    op.drop_index("ix_snapshots_source_fetched_at", table_name="snapshots")
    op.drop_table("snapshots")
    op.drop_index("ix_players_code", table_name="players")
    op.drop_table("players")
    op.drop_index("ix_fixtures_season_gameweek", table_name="fixtures")
    op.drop_table("fixtures")
