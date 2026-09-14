from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.db.tables import Snapshot, SnapshotFile
from plpd.pipelines.upload import upload


class FakeStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> str:
        self.data.setdefault(key, data)
        return f"gh://owner/repo/snapshots-backfill/{key.replace('/', '-')}"

    def get(self, uri: str) -> bytes:
        return self.data[uri]

    def exists(self, key: str) -> bool:
        return key in self.data


def seed(session: Session, rows: list[tuple[str, str, str]]) -> None:
    snapshot = Snapshot(
        source="fpl_core",
        season="2026-2027",
        fetched_at=datetime(2026, 9, 1, tzinfo=UTC),
        source_ref="main",
        content_hash="pull",
        row_count=0,
    )
    session.add(snapshot)
    session.flush()
    for name, digest, uri in rows:
        session.add(
            SnapshotFile(
                snapshot_id=snapshot.id,
                name=name,
                content_hash=digest,
                storage_uri=uri,
                row_count=1,
            )
        )
    session.flush()


def test_every_uri_is_rewritten(session: Session) -> None:
    source = FakeStore()
    source.data = {"/archive/teams.parquet": b"teams", "/archive/players.parquet": b"players"}
    seed(
        session,
        [
            ("teams", "aaa", "/archive/teams.parquet"),
            ("players", "bbb", "/archive/players.parquet"),
        ],
    )

    stored = upload(session, source, FakeStore())

    assert stored == 2
    uris = list(session.scalars(select(SnapshotFile.storage_uri)))
    assert all(uri.startswith("gh://") for uri in uris)


def test_one_hash_uploads_once_and_is_shared(session: Session) -> None:
    source = FakeStore()
    source.data = {"/archive/a.parquet": b"same", "/archive/b.parquet": b"same"}
    seed(session, [("teams", "aaa", "/archive/a.parquet"), ("teams2", "aaa", "/archive/b.parquet")])

    stored = upload(session, source, FakeStore())

    assert stored == 1
    assert len(set(session.scalars(select(SnapshotFile.storage_uri)))) == 1


def test_a_row_already_uploaded_is_left_alone(session: Session) -> None:
    source = FakeStore()
    source.data = {"/archive/teams.parquet": b"teams"}
    seed(
        session,
        [
            ("teams", "aaa", "gh://owner/repo/snapshots-backfill/fpl_core-teams-aaa.parquet"),
            ("players", "bbb", "/archive/teams.parquet"),
        ],
    )

    stored = upload(session, source, FakeStore())

    assert stored == 1


def test_the_key_carries_the_snapshot_source(session: Session) -> None:
    source = FakeStore()
    source.data = {"/archive/teams.parquet": b"teams"}
    seed(session, [("teams", "aaa", "/archive/teams.parquet")])
    target = FakeStore()

    upload(session, source, target)

    assert "fpl_core/teams/aaa.parquet" in target.data
