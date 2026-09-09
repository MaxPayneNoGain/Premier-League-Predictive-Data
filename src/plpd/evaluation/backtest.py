"""Split on kickoff time, not gameweek number. Upstream moves postponed games
into the week they're played, so the two match right now, but that could change.

A round is a season and a gameweek together, because gameweek numbers restart
every August.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plpd.evaluation.metrics import (
    Outcomes,
    Probabilities,
    brier_score,
    log_loss,
    ranked_probability_score,
)
from plpd.features import outcomes

Predictor = Callable[[pd.DataFrame, pd.DataFrame], Probabilities]


@dataclass(frozen=True)
class Fold:
    season: str
    gameweek: int
    train: pd.DataFrame
    test: pd.DataFrame


def walk_forward(matches: pd.DataFrame, *, min_train: int = 60) -> Iterator[Fold]:
    if min_train < 1:
        raise ValueError(f"min_train must be at least 1, got {min_train}")

    # Grouping on the gameweek number alone would put both seasons' round 8 in
    # one fold and cut the training window at the earlier of the two.
    rounds = (
        matches.groupby(["season", "gameweek"])["kickoff_time"]
        .min()
        .reset_index()
        .sort_values("kickoff_time")
    )

    for row in rounds.to_dict(orient="records"):
        season = str(row["season"])
        gameweek = int(row["gameweek"])
        test = matches[(matches["season"] == season) & (matches["gameweek"] == gameweek)]
        train = matches[matches["kickoff_time"] < row["kickoff_time"]]
        # 60 is about six rounds. Fewer than that and the strengths are noise.
        if len(train) < min_train:
            continue
        yield Fold(season=season, gameweek=gameweek, train=train, test=test)


@dataclass(frozen=True)
class Scores:
    matches: int
    brier: float
    log_loss: float
    rps: float


def pooled_predictions(
    matches: pd.DataFrame, predictor: Predictor, *, min_train: int = 60
) -> tuple[Probabilities, Outcomes]:
    predicted = []
    actual = []
    for fold in walk_forward(matches, min_train=min_train):
        predicted.append(predictor(fold.train, fold.test))
        actual.append(outcomes(fold.test))

    if not predicted:
        raise ValueError(f"no gameweek had {min_train} earlier matches to learn from")

    # Pooled rather than averaged per fold, so a week with 7 matches does not
    # weigh the same as one with 13.
    return np.vstack(predicted), np.concatenate(actual)


def score(probabilities: Probabilities, results: Outcomes) -> Scores:
    return Scores(
        matches=len(results),
        brier=brier_score(probabilities, results),
        log_loss=log_loss(probabilities, results),
        rps=ranked_probability_score(probabilities, results),
    )


def score_walk_forward(
    matches: pd.DataFrame, predictor: Predictor, *, min_train: int = 60
) -> Scores:
    return score(*pooled_predictions(matches, predictor, min_train=min_train))
