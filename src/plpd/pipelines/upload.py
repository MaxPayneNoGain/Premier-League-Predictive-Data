import argparse
import logging
import sys
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.db.tables import Snapshot, SnapshotFile
from plpd.ingest.storage import FilesystemStore, GitHubReleaseStore, ObjectStore

log = logging.getLogger("plpd.upload")

BACKFILL_TAG = "snapshots-backfill"


def upload(session: Session, source: ObjectStore, target: ObjectStore) -> int:
    rows = session.execute(
        select(SnapshotFile, Snapshot.source)
        .join(Snapshot, Snapshot.id == SnapshotFile.snapshot_id)
        .order_by(SnapshotFile.id)
    ).all()

    moved: dict[str, str] = {}
    stored = 0

    for row, origin in rows:
        if row.storage_uri.startswith("gh://"):
            continue

        uri = moved.get(row.content_hash)
        if uri is None:
            key = f"{origin}/{row.name}/{row.content_hash}.parquet"
            uri = target.put(key, source.get(row.storage_uri))
            moved[row.content_hash] = uri
            stored += 1
            log.info("stored %s", key)
        row.storage_uri = uri

    session.flush()
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-upload")
    parser.add_argument("--tag", default=BACKFILL_TAG)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    if not settings.archive_repo or not settings.archive_token:
        log.error("set PLPD_ARCHIVE_REPO and PLPD_ARCHIVE_TOKEN first")
        return 1

    target = GitHubReleaseStore(settings.archive_repo, settings.archive_token, tag=args.tag)

    with build_session_factory(build_engine(settings))() as session, session.begin():
        stored = upload(session, FilesystemStore(settings.snapshot_root), target)

    log.info("stored %d files", stored)
    return 0


if __name__ == "__main__":
    sys.exit(main())
