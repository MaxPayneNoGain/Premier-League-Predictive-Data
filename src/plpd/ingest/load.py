"""Promotion of an archived snapshot into the relational tables.

Archives keep every column as text so that repeated pulls stay comparable, which
makes this the place where each column's intended type is decided.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from plpd.db.tables import Player, Snapshot, Team
from plpd.db.upsert import upsert


def to_int(value: str) -> int | None:
    # A blank means upstream has no value for the column, which is not the same
    # as zero.
    return int(value) if value else None


def read_frame(snapshot: Snapshot, name: str) -> pd.DataFrame:
    path = Path(snapshot.storage_uri) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"snapshot {snapshot.id} has no {name}.parquet at {path}")
    return pd.read_parquet(path)


def load_teams(session: Session, snapshot: Snapshot) -> int:
    rows: list[dict[str, Any]] = [
        {
            "season": snapshot.season,
            "fpl_id": to_int(row["id"]),
            "code": to_int(row["code"]),
            "name": row["name"],
            "short_name": row["short_name"],
        }
        for row in read_frame(snapshot, "teams").to_dict(orient="records")
    ]
    upsert(session, Team, rows)
    return len(rows)


def load_players(session: Session, snapshot: Snapshot) -> int:
    rows: list[dict[str, Any]] = [
        {
            "season": snapshot.season,
            "fpl_id": to_int(row["player_id"]),
            "code": to_int(row["player_code"]),
            "first_name": row["first_name"],
            "second_name": row["second_name"],
            "web_name": row["web_name"],
            "team_code": to_int(row["team_code"]),
            "position": row["position"],
        }
        for row in read_frame(snapshot, "players").to_dict(orient="records")
    ]
    upsert(session, Player, rows)
    return len(rows)
