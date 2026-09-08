from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.db.tables import Player, Snapshot, Team
from plpd.ingest.load import load_players, load_teams

FETCHED_AT = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)

TEAMS = pd.DataFrame(
    [
        {"code": "3", "id": "1", "name": "Arsenal", "short_name": "ARS"},
        {"code": "91", "id": "2", "name": "AFC Bournemouth", "short_name": "BOU"},
    ]
)

PLAYERS = pd.DataFrame(
    [
        {
            "player_code": "208706",
            "player_id": "452",
            "first_name": "Bruno",
            "second_name": "Guimaraes",
            "web_name": "Bruno G.",
            "team_code": "4",
            "position": "Midfielder",
        }
    ]
)


def make_snapshot(
    session: Session, root: Path, frames: dict[str, pd.DataFrame], *, tag: str = "a"
) -> Snapshot:
    destination = root / tag
    destination.mkdir(parents=True)
    for name, frame in frames.items():
        frame.to_parquet(destination / f"{name}.parquet", index=False)

    snapshot = Snapshot(
        source="fpl_core",
        season="2026-2027",
        fetched_at=FETCHED_AT,
        source_ref="main",
        storage_uri=destination.as_posix(),
        content_hash=tag * 64,
        row_count=sum(len(frame) for frame in frames.values()),
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def test_teams_load_with_the_snapshot_season(session: Session, tmp_path: Path) -> None:
    snapshot = make_snapshot(session, tmp_path, {"teams": TEAMS})

    assert load_teams(session, snapshot) == 2

    arsenal = session.get(Team, ("2026-2027", 1))
    assert arsenal is not None
    assert (arsenal.name, arsenal.code) == ("Arsenal", 3)


def test_players_keep_id_and_code_apart(session: Session, tmp_path: Path) -> None:
    snapshot = make_snapshot(session, tmp_path, {"players": PLAYERS})

    load_players(session, snapshot)

    player = session.get(Player, ("2026-2027", 452))
    assert player is not None
    assert player.code == 208706
    assert player.team_code == 4


def test_a_later_snapshot_overwrites_rather_than_duplicates(
    session: Session, tmp_path: Path
) -> None:
    load_teams(session, make_snapshot(session, tmp_path, {"teams": TEAMS}))

    renamed = TEAMS.copy()
    renamed.loc[0, "short_name"] = "ARE"
    load_teams(session, make_snapshot(session, tmp_path, {"teams": renamed}, tag="b"))

    assert len(session.scalars(select(Team)).all()) == 2
    arsenal = session.get(Team, ("2026-2027", 1))
    assert arsenal is not None
    assert arsenal.short_name == "ARE"


def test_a_missing_file_names_the_snapshot_and_the_path(session: Session, tmp_path: Path) -> None:
    snapshot = make_snapshot(session, tmp_path, {"teams": TEAMS})

    with pytest.raises(FileNotFoundError, match=r"players\.parquet"):
        load_players(session, snapshot)
