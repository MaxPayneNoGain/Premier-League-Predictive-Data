"""Index the archives taken before snapshots recorded their files.

A one-off. Every pull made from here on writes its own index rows.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.db.tables import Snapshot, SnapshotFile
from plpd.ingest.snapshots import file_hash

log = logging.getLogger("plpd.reindex")


def reindex(session: Session) -> int:
    indexed = set(session.scalars(select(SnapshotFile.snapshot_id)))
    written = 0

    for snapshot in session.scalars(select(Snapshot).order_by(Snapshot.id)):
        if snapshot.id in indexed:
            continue

        directory = Path(snapshot.storage_uri)
        if not directory.is_dir():
            log.warning("snapshot %d: %s is gone", snapshot.id, snapshot.storage_uri)
            continue

        for path in sorted(directory.glob("*.parquet")):
            session.add(
                SnapshotFile(
                    snapshot_id=snapshot.id,
                    name=path.stem,
                    content_hash=file_hash(path.read_bytes()),
                    storage_uri=path.as_posix(),
                    row_count=len(pd.read_parquet(path)),
                )
            )
            written += 1

    session.flush()
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-reindex")
    parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    with build_session_factory(build_engine(Settings()))() as session, session.begin():
        written = reindex(session)

    log.info("indexed %d archived files", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
