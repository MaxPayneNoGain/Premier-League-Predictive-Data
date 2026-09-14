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
    Scores,
    always_home,
    base_rates,
    decompose,
    devig,
    outcome_rates,
    overround,
    pooled_predictions,
    prices,
    reliability_bins,
    score,
    uniform,
)
from plpd.features import PRICE_COLUMNS, finished_matches, match_odds, outcomes
from plpd.ingest.odds import BOOKMAKER
from plpd.models import HALF_LIFE_DAYS, fit, predict
from plpd.predictions import MODEL, probabilities

log = logging.getLogger("plpd.backtest")

NAIVE = "base rates"

BEST = MODEL

PREDICTORS: dict[str, Predictor] = {
    "uniform": lambda _, test: uniform(len(test)),
    "always home": lambda _, test: always_home(len(test)),
    NAIVE: lambda train, test: base_rates(outcomes(train), len(test)),
    "poisson": lambda train, test: predict(fit(train), test),
    "dixon-coles": lambda train, test: predict(fit(train, correlation=True), test),
    "dixon-coles decayed": lambda train, test: predict(
        fit(train, correlation=True, half_life=HALF_LIFE_DAYS), test
    ),
    BEST: probabilities,
}

MARKET = f"{BOOKMAKER} de-vigged"

OUTCOMES = ("home", "draw", "away")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plpd-backtest")
    parser.add_argument("--season", action="append", metavar="YYYY-YYYY")
    parser.add_argument("--tournament", default="prem")
    parser.add_argument("--min-train", type=int, default=60)
    parser.add_argument(
        "--calibration",
        metavar="PREDICTOR",
        help="also print the reliability table for one predictor",
    )
    args = parser.parse_args(argv)

    known = [*PREDICTORS, MARKET]
    if args.calibration is not None and args.calibration not in known:
        parser.error(f"unknown predictor {args.calibration!r}, pick from {', '.join(known)}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    settings = Settings()
    with build_session_factory(build_engine(settings))() as session:
        matches = finished_matches(session, tournament=args.tournament, seasons=args.season)
        odds = match_odds(session, tournament=args.tournament)

    if matches.empty:
        window = " in " + ", ".join(args.season) if args.season else ""
        log.error(
            "no finished %s matches%s are loaded - run plpd-load first", args.tournament, window
        )
        return 1

    matches = matches.merge(odds, on="match_id", how="left")
    priced = int(matches[PRICE_COLUMNS].notna().all(axis=1).sum())
    predictors = dict(PREDICTORS)
    if priced == len(matches):
        predictors[MARKET] = lambda _, test: devig(prices(test))
    elif priced:
        # Scoring the market on the matches it happens to cover would put a row
        # in the table that was not measured on the same fixtures as the rest.
        log.warning("odds cover %d of %d matches, leaving the market out", priced, len(matches))
    else:
        log.warning("no odds are loaded - run plpd-load --odds to score the market")

    chosen = None
    uncertainty = 0.0
    scored: dict[str, Scores] = {}
    if args.season:
        print(f"seasons {', '.join(args.season)}\n")
    print(
        f"{'predictor':<22}{'matches':>9}{'Brier':>9}{'log loss':>11}"
        f"{'RPS':>9}{'reliability':>13}{'resolution':>12}"
    )
    for name, predictor in predictors.items():
        probabilities, results = pooled_predictions(matches, predictor, min_train=args.min_train)
        scores = score(probabilities, results)
        scored[name] = scores
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

    if MARKET in predictors:
        margin = overround(prices(matches)).mean()
        print(f"bookmaker margin {margin:.2%} on average, taken out before scoring")
        _print_gap(scored)

    if chosen is not None:
        _print_calibration(*chosen, name=str(args.calibration))

    return 0


def _print_gap(scored: dict[str, Scores]) -> None:
    naive, model, market = scored[NAIVE], scored[BEST], scored[MARKET]
    if naive.rps <= market.rps:
        return

    print(f"\nshare of the distance from {NAIVE} to {MARKET} that {BEST} covers")
    metrics = (
        ("Brier", naive.brier, model.brier, market.brier),
        ("log loss", naive.log_loss, model.log_loss, market.log_loss),
        ("RPS", naive.rps, model.rps, market.rps),
    )
    for label, floor, reached, target in metrics:
        print(f"{label:<14}{(floor - reached) / (floor - target):>10.0%}")


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
