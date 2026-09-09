"""Score every predictor walk-forward and print the table."""

import argparse
import logging
import sys
from collections.abc import Sequence

from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.evaluation import (
    Outcomes,
    Predictor,
    Probabilities,
    always_home,
    base_rates,
    decompose,
    outcome_rates,
    pooled_predictions,
    reliability_bins,
    score,
    uniform,
)
from plpd.features import finished_matches, outcomes
from plpd.models import HALF_LIFE_DAYS, fit, predict

log = logging.getLogger("plpd.backtest")

PREDICTORS: dict[str, Predictor] = {
    "uniform": lambda _, test: uniform(len(test)),
    "always home": lambda _, test: always_home(len(test)),
    "base rates": lambda train, test: base_rates(outcomes(train), len(test)),
    "poisson": lambda train, test: predict(fit(train), test),
    "dixon-coles": lambda train, test: predict(fit(train, correlation=True), test),
    "dixon-coles decayed": lambda train, test: predict(
        fit(train, correlation=True, half_life=HALF_LIFE_DAYS), test
    ),
}

OUTCOMES = ("home", "draw", "away")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-backtest")
    parser.add_argument("--tournament", default="prem")
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument(
        "--calibration",
        metavar="PREDICTOR",
        help="also print the reliability table for one predictor",
    )
    args = parser.parse_args(argv)

    if args.calibration is not None and args.calibration not in PREDICTORS:
        parser.error(f"unknown predictor {args.calibration!r}, pick from {', '.join(PREDICTORS)}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    with build_session_factory(build_engine(settings))() as session:
        matches = finished_matches(session, tournament=args.tournament)

    if matches.empty:
        log.error("no finished %s matches are loaded - run plpd-load first", args.tournament)
        return 1

    chosen = None
    uncertainty = 0.0
    print(
        f"{'predictor':<22}{'matches':>9}{'Brier':>9}{'log loss':>11}"
        f"{'RPS':>9}{'reliability':>13}{'resolution':>12}"
    )
    for name, predictor in PREDICTORS.items():
        probabilities, results = pooled_predictions(matches, predictor, min_train=args.min_train)
        scores = score(probabilities, results)
        parts = decompose(probabilities, results)
        uncertainty = parts.uncertainty
        print(
            f"{name:<22}{scores.matches:>9}{scores.brier:>9.4f}"
            f"{scores.log_loss:>11.4f}{scores.rps:>9.4f}"
            f"{parts.reliability:>13.4f}{parts.resolution:>12.4f}"
        )
        if name == args.calibration:
            chosen = (probabilities, results)

    print(f"\nuncertainty {uncertainty:.4f}, set by the results alone and equal for every row")

    if chosen is not None:
        _print_calibration(*chosen, name=str(args.calibration))

    return 0


def _print_calibration(probabilities: Probabilities, results: Outcomes, *, name: str) -> None:
    print(f"\nreliability for {name}")
    print(f"{'bin':<14}{'forecasts':>11}{'predicted':>11}{'observed':>10}")
    for slot in reliability_bins(probabilities, results):
        if not slot.count:
            continue
        label = f"{slot.lower:.1f} to {slot.upper:.1f}"
        print(f"{label:<14}{slot.count:>11}{slot.predicted:>11.4f}{slot.observed:>10.4f}")

    print(f"\n{'outcome':<14}{'predicted':>11}{'observed':>10}")
    rates = outcome_rates(probabilities, results)
    for outcome, (predicted, observed) in zip(OUTCOMES, rates, strict=True):
        print(f"{outcome:<14}{predicted:>11.4f}{observed:>10.4f}")


if __name__ == "__main__":
    sys.exit(main())
