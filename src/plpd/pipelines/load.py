"""Promote the most recent archived pull into the tables.

Run after plpd-snapshot. Loading is separate from fetching so that a rerun costs
nothing upstream and so a bad cast can be fixed and replayed against archives
that already exist.
"""

import argparse
import logging
import sys
from collections.abc import Sequence

from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.ingest import FplCoreSource
from plpd.ingest.load import (
    fixture_paths,
    latest_snapshot,
    load_fixtures,
    load_players,
    load_teams,
)

log = logging.getLogger("plpd.load")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-load")
    parser.add_argument("--season", action="append", metavar="YYYY-YYYY")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    session_factory = build_session_factory(build_engine(settings))

    for season in args.season or [settings.current_season]:
        with session_factory() as session, session.begin():
            snapshot = latest_snapshot(session, source=FplCoreSource.name, season=season)
            if snapshot is None:
                log.error("no snapshot archived for %s - run plpd-snapshot first", season)
                return 1

            teams = load_teams(session, snapshot)
            players = load_players(session, snapshot)
            fixtures = load_fixtures(session, snapshot) if fixture_paths(snapshot) else 0
            log.info(
                "snapshot %d: %d teams, %d players, %d fixtures",
                snapshot.id,
                teams,
                players,
                fixtures,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
