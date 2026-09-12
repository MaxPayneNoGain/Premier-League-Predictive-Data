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
from plpd.ingest import ClubFootballOddsSource, FplCoreLegacySource, FplCoreSource
from plpd.ingest.load import (
    archived_snapshots,
    fixture_names,
    has_frame,
    latest_snapshot,
    load_fixtures,
    load_matches,
    load_odds,
    load_players,
    load_teams,
)

log = logging.getLogger("plpd.load")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-load")
    parser.add_argument("--season", action="append", metavar="YYYY-YYYY")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Replay every archived pull for the season, oldest first.",
    )
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument(
        "--legacy",
        action="store_true",
        help="Load whole-season snapshots taken with plpd-snapshot --legacy.",
    )
    layout.add_argument(
        "--odds",
        action="store_true",
        help="Load bookmaker prices taken with plpd-snapshot --odds.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    session_factory = build_session_factory(build_engine(settings))
    if args.odds:
        source = ClubFootballOddsSource.name
    elif args.legacy:
        source = FplCoreLegacySource.name
    else:
        source = FplCoreSource.name

    for season in args.season or [settings.current_season]:
        with session_factory() as session, session.begin():
            if args.all:
                snapshots = archived_snapshots(session, source=source, season=season)
            else:
                newest = latest_snapshot(session, source=source, season=season)
                snapshots = [newest] if newest is not None else []

            if not snapshots:
                log.error("no snapshot archived for %s - run plpd-snapshot first", season)
                return 1

            for snapshot in snapshots:
                if args.odds:
                    log.info("snapshot %d: %d odds", snapshot.id, load_odds(session, snapshot))
                    continue

                if args.legacy:
                    # The earliest legacy snapshots archived matches only.
                    legacy_teams = (
                        load_teams(session, snapshot)
                        if has_frame(session, snapshot, "teams")
                        else 0
                    )
                    log.info(
                        "snapshot %d: %d teams, %d matches",
                        snapshot.id,
                        legacy_teams,
                        load_matches(session, snapshot),
                    )
                    continue

                teams = load_teams(session, snapshot)
                players = load_players(session, snapshot)
                has_fixtures = bool(fixture_names(session, snapshot))
                fixtures = load_fixtures(session, snapshot) if has_fixtures else 0
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
