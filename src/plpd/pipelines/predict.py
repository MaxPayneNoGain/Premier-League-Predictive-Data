"""Predict one matchweek and record what was predicted.

The training window is cut at the round's first kickoff, so a gameweek that has
already been played is fitted on what existed before it.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd

from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.features import finished_matches, next_round, round_fixtures
from plpd.predictions import predict_round, record

log = logging.getLogger("plpd.predict")

MIN_TRAIN = 60


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-predict")
    parser.add_argument("--season", metavar="YYYY-YYYY")
    parser.add_argument("--gameweek", type=int, help="Defaults to the next round to be played.")
    parser.add_argument("--tournament", default="prem")
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the round without recording it.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    season = args.season or settings.current_season

    with build_session_factory(build_engine(settings))() as session, session.begin():
        gameweek = args.gameweek or next_round(session, season=season, tournament=args.tournament)
        if gameweek is None:
            log.error(
                "every %s fixture in %s is played - name a gameweek to predict",
                season,
                args.tournament,
            )
            return 1

        fixtures = round_fixtures(
            session, season=season, gameweek=gameweek, tournament=args.tournament
        )
        if fixtures.empty:
            log.error(
                "no %s fixtures loaded for %s gameweek %d - run plpd-load first",
                args.tournament,
                season,
                gameweek,
            )
            return 1

        kickoff = fixtures["kickoff_time"].min()
        if pd.isna(kickoff):
            log.error("%s gameweek %d has no kickoff time yet", season, gameweek)
            return 1

        history = finished_matches(session, tournament=args.tournament)
        history = history[history["kickoff_time"] < kickoff]
        if len(history) < args.min_train:
            log.error(
                "only %d matches finished before %s, need %d to fit",
                len(history),
                kickoff,
                args.min_train,
            )
            return 1

        predictions = predict_round(history, fixtures)
        predicted_at = datetime.now(UTC)
        if predicted_at > kickoff:
            log.info("gameweek %d has already kicked off, so this is a retrodiction", gameweek)

        _print_round(predictions, season=season, gameweek=gameweek, trained_on=len(history))

        if args.dry_run:
            return 0

        written = record(
            session,
            predictions,
            predicted_at=predicted_at,
            trained_through=history["kickoff_time"].max(),
        )
        log.info("recorded %d predictions for %s gameweek %d", written, season, gameweek)

    return 0


def _print_round(predictions: pd.DataFrame, *, season: str, gameweek: int, trained_on: int) -> None:
    print(f"\n{season} gameweek {gameweek}, fitted on {trained_on} earlier matches\n")
    print(f"{'match':<46}{'home':>8}{'draw':>8}{'away':>8}{'home xG':>10}{'away xG':>10}")
    for row in predictions.to_dict(orient="records"):
        print(
            f"{row['match_id']!s:<46}{row['home_win']:>8.4f}{row['draw']:>8.4f}"
            f"{row['away_win']:>8.4f}{row['home_xg']:>10.2f}{row['away_xg']:>10.2f}"
        )


if __name__ == "__main__":
    sys.exit(main())
