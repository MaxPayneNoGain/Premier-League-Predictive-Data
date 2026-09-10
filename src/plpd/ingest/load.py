"""Promotion of an archived snapshot into the relational tables.

Archives keep every column as text so that repeated pulls stay comparable, which
makes this the place where each column's intended type is decided. Every cast
here treats a blank as absent rather than as a zero-valued default.
"""

from collections.abc import Hashable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.db.tables import Fixture, Player, Snapshot, Team
from plpd.db.upsert import upsert

# 2024-2025 writes these uppercase, later seasons write them capitalised.
BOOLEANS = {"True": True, "TRUE": True, "False": False, "FALSE": False}

TOURNAMENTS = ("champions-league", "conference-league", "efl-cup", "europa-league", "prem")


def to_int(value: str) -> int | None:
    if not value:
        return None
    # Whole numbers arrive float-formatted ("88.0") wherever upstream writes the
    # column as a float, and int() rejects that string outright. Going through
    # float() accepts both spellings; refusing a fractional value keeps it from
    # quietly truncating an id into a different club's.
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"expected a whole number, got {value!r}")
    return int(number)


def to_float(value: str) -> float | None:
    return float(value) if value else None


def to_bool(value: str) -> bool:
    # bool("False") is True, so the spellings have to be mapped rather than
    # coerced - otherwise every unplayed fixture loads as finished.
    if value not in BOOLEANS:
        raise ValueError(f"expected a boolean spelling, got {value!r}")
    return BOOLEANS[value]


def to_datetime(value: str) -> datetime | None:
    """Read a kickoff time, which upstream writes without an offset.

    The Saturday slots in the data land on 11:30, 14:00 and 16:30, which are the
    league's 12:30, 15:00 and 17:30 British Summer Time kickoffs. So the clock is
    UTC and only the marker is missing.
    """
    return datetime.fromisoformat(value).replace(tzinfo=UTC) if value else None


def tournament_of(match_id: str, season: str) -> str:
    """Recover the competition from a match id, for files with no such column.

    Ids read `24-25-prem-afc-bournemouth-vs-arsenal`, so the season is a two by
    two digit prefix and the competition follows it.
    """
    prefix = f"{season[2:4]}-{season[7:]}-"
    if not match_id.startswith(prefix):
        raise ValueError(f"{match_id!r} does not belong to season {season}")

    rest = match_id.removeprefix(prefix)
    for tournament in TOURNAMENTS:
        if rest.startswith(f"{tournament}-"):
            return tournament
    raise ValueError(f"no known competition in {match_id!r}")


def latest_snapshot(session: Session, *, source: str, season: str) -> Snapshot | None:
    return session.scalar(
        select(Snapshot)
        .where(Snapshot.source == source, Snapshot.season == season)
        .order_by(Snapshot.fetched_at.desc())
    )


def archived_snapshots(session: Session, *, source: str, season: str) -> list[Snapshot]:
    """Every pull for a season, oldest first.

    A backfill archives one gameweek at a time, so a season's fixtures are
    spread across as many snapshots as there are gameweeks and only a replay
    reaches them all. Oldest first means the newest pull is the one that lands
    last and wins.
    """
    return list(
        session.scalars(
            select(Snapshot)
            .where(Snapshot.source == source, Snapshot.season == season)
            .order_by(Snapshot.fetched_at)
        )
    )


def fixture_paths(snapshot: Snapshot) -> list[Path]:
    # A pull made without --gameweek archives the season files only, so callers
    # need to be able to ask before loading rather than handle a failure.
    return sorted(Path(snapshot.storage_uri).glob("GW*__fixtures.parquet"))


def has_frame(snapshot: Snapshot, name: str) -> bool:
    return (Path(snapshot.storage_uri) / f"{name}.parquet").exists()


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


def _fixture_row(row: Mapping[Hashable, Any], *, season: str, tournament: str) -> dict[str, Any]:
    return {
        "match_id": row["match_id"],
        "season": season,
        "gameweek": to_int(row["gameweek"]),
        "tournament": tournament,
        "kickoff_time": to_datetime(row["kickoff_time"]),
        "home_team_code": to_int(row["home_team"]),
        "away_team_code": to_int(row["away_team"]),
        "home_team_elo": to_float(row["home_team_elo"]),
        "away_team_elo": to_float(row["away_team_elo"]),
        "home_score": to_int(row["home_score"]),
        "away_score": to_int(row["away_score"]),
        "finished": to_bool(row["finished"]),
    }


def load_fixtures(session: Session, snapshot: Snapshot) -> int:
    """Load every gameweek's fixtures the archive happens to hold.

    The team columns carry codes rather than FPL ids, and the file has no season
    column, so that comes from the snapshot.
    """
    paths = fixture_paths(snapshot)
    if not paths:
        raise FileNotFoundError(
            f"snapshot {snapshot.id} archived no gameweek fixtures under {snapshot.storage_uri}"
        )

    rows: list[dict[str, Any]] = [
        _fixture_row(row, season=snapshot.season, tournament=row["tournament"])
        for path in paths
        for row in pd.read_parquet(path).to_dict(orient="records")
    ]
    upsert(session, Fixture, rows)
    return len(rows)


def load_matches(session: Session, snapshot: Snapshot) -> int:
    """Load a whole season from the single matches file the older layout ships.

    That file carries no `tournament` column, so the competition is read back
    out of the match id.
    """
    rows: list[dict[str, Any]] = [
        _fixture_row(
            row,
            season=snapshot.season,
            tournament=tournament_of(row["match_id"], snapshot.season),
        )
        for row in read_frame(snapshot, "matches").to_dict(orient="records")
    ]
    upsert(session, Fixture, rows)
    return len(rows)
