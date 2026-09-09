"""Fetch one pull of the upstream dataset and archive it.

Run after upstream refreshes at 07:30 and 17:30 UTC.
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.ingest import FplCoreLegacySource, FplCoreSource, SourceError, archive

log = logging.getLogger("plpd.snapshot")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-snapshot")
    parser.add_argument("--season", action="append", metavar="YYYY-YYYY")
    parser.add_argument("--gameweek", type=int)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Read the older layout, one whole-season file instead of gameweek pulls.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    source = FplCoreLegacySource(settings) if args.legacy else FplCoreSource(settings)
    session_factory = build_session_factory(build_engine(settings))

    for season in args.season or [settings.current_season]:
        try:
            files = source.fetch(season, gameweek=args.gameweek)
        except SourceError as exc:
            log.error("%s", exc)
            return 1

        with session_factory() as session, session.begin():
            snapshot = archive(
                session,
                files,
                source=source.name,
                season=season,
                source_ref=settings.fpl_core_ref,
                root=settings.snapshot_root,
            )
            if snapshot is None:
                log.info("%s unchanged since last pull", season)
            else:
                log.info("archived %d files to %s", len(files), snapshot.storage_uri)

    return 0


if __name__ == "__main__":
    sys.exit(main())
