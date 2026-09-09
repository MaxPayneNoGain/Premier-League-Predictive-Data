"""Split on kickoff time, not gameweek number. Upstream moves postponed games
into the week they're played, so the two match right now, but that could change.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


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
