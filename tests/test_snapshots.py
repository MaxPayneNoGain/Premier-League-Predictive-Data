from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from plpd.db.tables import Snapshot, SnapshotFile
from plpd.ingest import SourceFile, archive, content_hash
from plpd.ingest.fpl_core import read_csv

FETCHED_AT = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)


def make_file(name: str, raw: bytes) -> SourceFile:
    return SourceFile(name=name, raw=raw, frame=read_csv(raw))


@pytest.fixture
def files() -> list[SourceFile]:
    return [
        make_file("players.csv", b"player_code,web_name\n118748,Vieira\n"),
        make_file("GW3/fixtures.csv", b"gameweek,match_id\n3,abc\n"),
    ]


def test_content_hash_ignores_file_order(files: list[SourceFile]) -> None:
    assert content_hash(files) == content_hash(list(reversed(files)))


def test_content_hash_changes_with_content(files: list[SourceFile]) -> None:
    changed = [make_file("players.csv", b"player_code,web_name\n118748,Saka\n"), files[1]]
    assert content_hash(files) != content_hash(changed)


def test_archive_writes_parquet_and_flattens_nested_names(
    session: Session, files: list[SourceFile], tmp_path: Path
) -> None:
    snapshot = archive(
        session,
        files,
        source="fpl_core",
        season="2026-2027",
        source_ref="main",
        root=tmp_path,
        fetched_at=FETCHED_AT,
    )

    assert snapshot is not None
    written = sorted(p.name for p in Path(snapshot.storage_uri).glob("*.parquet"))
    assert written == ["GW3__fixtures.parquet", "players.parquet"]
    assert len(pd.read_parquet(Path(snapshot.storage_uri) / "players.parquet")) == 1
    assert snapshot.row_count == 2


def test_identical_pull_is_skipped(
    session: Session, files: list[SourceFile], tmp_path: Path
) -> None:
    def run(fetched_at: datetime) -> Snapshot | None:
        return archive(
            session,
            files,
            source="fpl_core",
            season="2026-2027",
            source_ref="main",
            root=tmp_path,
            fetched_at=fetched_at,
        )

    first = run(FETCHED_AT)
    second = run(datetime(2026, 8, 31, 18, 0, tzinfo=UTC))

    assert first is not None
    assert second is None
    assert len(list(tmp_path.rglob("*.parquet"))) == 2


def test_empty_pull_is_rejected(session: Session, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to archive"):
        archive(
            session, [], source="fpl_core", season="2026-2027", source_ref="main", root=tmp_path
        )


def make_snapshot(session: Session) -> Snapshot:
    snapshot = Snapshot(
        source="fpl_core",
        season="2026-2027",
        fetched_at=FETCHED_AT,
        source_ref="main",
        storage_uri="unused",
        content_hash="a" * 64,
        row_count=1,
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def test_snapshot_file_rows_belong_to_a_snapshot(session: Session) -> None:
    snapshot = make_snapshot(session)

    session.add(
        SnapshotFile(
            snapshot_id=snapshot.id,
            name="players",
            content_hash="b" * 64,
            storage_uri="fpl_core/players/bbbb.parquet",
            row_count=1,
        )
    )
    session.flush()

    stored = session.scalars(select(SnapshotFile)).one()
    assert stored.snapshot_id == snapshot.id
    assert stored.name == "players"


def test_one_row_per_file_name_in_a_snapshot(session: Session) -> None:
    snapshot = make_snapshot(session)

    for digest in ("b" * 64, "c" * 64):
        session.add(
            SnapshotFile(
                snapshot_id=snapshot.id,
                name="players",
                content_hash=digest,
                storage_uri=f"fpl_core/players/{digest}.parquet",
                row_count=1,
            )
        )

    with pytest.raises(IntegrityError):
        session.flush()
