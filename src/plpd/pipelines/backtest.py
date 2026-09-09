"""Score the baselines walk-forward and print the table."""

import argparse
import logging
import sys
from collections.abc import Sequence

from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.evaluation import (
    Predictor,
    always_home,
    base_rates,
    score_walk_forward,
    uniform,
)
from plpd.features import finished_matches, outcomes

log = logging.getLogger("plpd.backtest")

BASELINES: dict[str, Predictor] = {
    "uniform": lambda _, test: uniform(len(test)),
    "always home": lambda _, test: always_home(len(test)),
    "base rates": lambda train, test: base_rates(outcomes(train), len(test)),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-backtest")
    parser.add_argument("--tournament", default="prem")
    parser.add_argument("--min-train", type=int, default=60)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    with build_session_factory(build_engine(settings))() as session:
        matches = finished_matches(session, tournament=args.tournament)

    if matches.empty:
        log.error("no finished %s matches are loaded - run plpd-load first", args.tournament)
        return 1

    print(f"{'baseline':<14}{'matches':>9}{'Brier':>9}{'log loss':>11}{'RPS':>9}")
    for name, predictor in BASELINES.items():
        scores = score_walk_forward(matches, predictor, min_train=args.min_train)
        print(
            f"{name:<14}{scores.matches:>9}{scores.brier:>9.4f}"
            f"{scores.log_loss:>11.4f}{scores.rps:>9.4f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
