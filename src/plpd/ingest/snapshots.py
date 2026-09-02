"""Archiving of raw source pulls.

Upstream overwrites its CSVs in place, so no history exists unless we keep our
own. Everything downstream reads these archives rather than the network.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.db.tables import Snapshot
from plpd.ingest.fpl_core import SourceFile


def content_hash(files: list[SourceFile]) -> str:
    # Sorted by name so the same pull hashes the same regardless of fetch order.
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda f: f.name):
        digest.update(file.name.encode())
        digest.update(file.raw)
    return digest.hexdigest()


def archive(
    session: Session,
    files: list[SourceFile],
    *,
    source: str,
    season: str,
    source_ref: str,
    root: Path,
    fetched_at: datetime | None = None,
) -> Snapshot | None:
    """Write the pull to Parquet and record it. None means upstream hadn't changed."""
    if not files:
        raise ValueError("nothing to archive - the source returned no files")

    digest = content_hash(files)
    seen = session.scalar(
        select(Snapshot).where(
            Snapshot.source == source,
            Snapshot.season == season,
            Snapshot.content_hash == digest,
        )
    )
    # The job runs twice a day but upstream often hasn't moved, and re-archiving
    # a byte-identical pull would grow the store for nothing.
    if seen is not None:
        return None

    fetched_at = fetched_at or datetime.now(UTC)
    # one directory per pull, named by the time we fetched it
    destination = root / source / season / fetched_at.strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=True)

    for file in files:
        stem = file.name.replace("/", "__").removesuffix(".csv")
        file.frame.to_parquet(destination / f"{stem}.parquet", index=False)

    snapshot = Snapshot(
        source=source,
        season=season,
        fetched_at=fetched_at,
        source_ref=source_ref,
        storage_uri=destination.as_posix(),
        content_hash=digest,
        row_count=sum(len(f.frame) for f in files),
    )
    session.add(snapshot)
    session.flush()
    return snapshot
