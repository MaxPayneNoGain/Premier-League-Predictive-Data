from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.db.tables import Snapshot, SnapshotFile
from plpd.pipelines.reindex import reindex

FETCHED_AT = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)


def make_snapshot(session: Session, root: Path, tag: str) -> Snapshot:
    destination = root / "fpl_core" / "2026-2027" / tag
    destination.mkdir(parents=True)
    pd.DataFrame({"code": ["3"]}).to_parquet(destination / "teams.parquet", index=False)
    pd.DataFrame({"gameweek": ["3"]}).to_parquet(destination / "GW3__fixtures.parquet", index=False)

    snapshot = Snapshot(
        source="fpl_core",
        season="2026-2027",
        fetched_at=FETCHED_AT,
        source_ref="main",
        storage_uri=destination.as_posix(),
        content_hash=tag.ljust(64, "0"),
        row_count=2,
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def test_reindex_writes_a_row_per_archived_file(session: Session, tmp_path: Path) -> None:
    snapshot = make_snapshot(session, tmp_path, "a")

    assert reindex(session) == 2

    rows = session.scalars(
        select(SnapshotFile).where(SnapshotFile.snapshot_id == snapshot.id)
    ).all()
    assert sorted(r.name for r in rows) == ["GW3__fixtures", "teams"]
    assert all(Path(r.storage_uri).exists() for r in rows)
    assert all(r.row_count == 1 for r in rows)


def test_reindex_skips_snapshots_already_indexed(session: Session, tmp_path: Path) -> None:
    make_snapshot(session, tmp_path, "a")

    assert reindex(session) == 2
    assert reindex(session) == 0


def test_identical_files_share_a_hash(session: Session, tmp_path: Path) -> None:
    make_snapshot(session, tmp_path, "a")
    make_snapshot(session, tmp_path, "b")

    reindex(session)

    hashes = session.scalars(
        select(SnapshotFile.content_hash).where(SnapshotFile.name == "teams")
    ).all()
    assert len(hashes) == 2
    assert len(set(hashes)) == 1


def test_a_snapshot_whose_directory_is_gone_is_skipped(session: Session, tmp_path: Path) -> None:
    make_snapshot(session, tmp_path, "a")
    session.add(
        Snapshot(
            source="fpl_core",
            season="2026-2027",
            fetched_at=FETCHED_AT,
            source_ref="main",
            storage_uri=(tmp_path / "gone").as_posix(),
            content_hash="c".ljust(64, "0"),
            row_count=0,
        )
    )
    session.flush()

    assert reindex(session) == 2
