"""Split on kickoff time, not gameweek number. Upstream moves postponed games
into the week they're played, so the two match right now, but that could change.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plpd.evaluation.metrics import (
    Probabilities,
    brier_score,
    log_loss,
    ranked_probability_score,
)
from plpd.features import outcomes

Predictor = Callable[[pd.DataFrame, pd.DataFrame], Probabilities]


@dataclass(frozen=True)
class Fold:
    gameweek: int
    train: pd.DataFrame
    test: pd.DataFrame


def walk_forward(matches: pd.DataFrame, *, min_train: int = 60) -> Iterator[Fold]:
    if min_train < 1:
        raise ValueError(f"min_train must be at least 1, got {min_train}")

    for gameweek in sorted(matches["gameweek"].unique()):
        test = matches[matches["gameweek"] == gameweek]
        cutoff = test["kickoff_time"].min()
        train = matches[matches["kickoff_time"] < cutoff]
        # 60 is about six rounds. Fewer than that and the strengths are noise.
        if len(train) < min_train:
            continue
        yield Fold(gameweek=int(gameweek), train=train, test=test)


@dataclass(frozen=True)
class Scores:
    matches: int
    brier: float
    log_loss: float
    rps: float


def score_walk_forward(
    matches: pd.DataFrame, predictor: Predictor, *, min_train: int = 60
) -> Scores:
    predicted = []
    actual = []
    for fold in walk_forward(matches, min_train=min_train):
        predicted.append(predictor(fold.train, fold.test))
        actual.append(outcomes(fold.test))

    if not predicted:
        raise ValueError(f"no gameweek had {min_train} earlier matches to learn from")

    # Pooled rather than averaged per fold, so a week with 7 matches does not
    # weigh the same as one with 13.
    probabilities = np.vstack(predicted)
    results = np.concatenate(actual)
    return Scores(
        matches=len(results),
        brier=brier_score(probabilities, results),
        log_loss=log_loss(probabilities, results),
        rps=ranked_probability_score(probabilities, results),
    )
