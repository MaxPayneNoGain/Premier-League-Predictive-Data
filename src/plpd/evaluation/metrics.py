"""Scoring rules for probabilistic match predictions.

All three are proper scoring rules, so none of them can be improved by reporting
anything other than an honest belief. Lower is better in every case.

Probabilities arrive as an (n, 3) array whose columns are ordered home, draw,
away; outcomes as an (n,) array of the index that actually happened.
"""

import numpy as np
import numpy.typing as npt

Probabilities = npt.NDArray[np.float64]
Outcomes = npt.NDArray[np.int_]

HOME, DRAW, AWAY = 0, 1, 2
CLASSES = 3

# Below this a single confident miss would dominate a whole season's log loss,
# and an outright zero would make it infinite.
FLOOR = 1e-15


def _checked(probabilities: Probabilities, outcomes: Outcomes) -> Probabilities:
    if probabilities.ndim != 2 or probabilities.shape[1] != CLASSES:
        raise ValueError(f"expected an (n, {CLASSES}) array, got {probabilities.shape}")
    if probabilities.shape[0] != outcomes.shape[0]:
        raise ValueError(
            f"{probabilities.shape[0]} predictions against {outcomes.shape[0]} outcomes"
        )
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("every row of probabilities must sum to 1")
    return probabilities


def _one_hot(outcomes: Outcomes) -> Probabilities:
    encoded = np.zeros((outcomes.shape[0], CLASSES))
    encoded[np.arange(outcomes.shape[0]), outcomes] = 1.0
    return encoded


def brier_score(probabilities: Probabilities, outcomes: Outcomes) -> float:
    """Mean squared error across all three outcomes. Ranges 0 to 2."""
    _checked(probabilities, outcomes)
    return float(np.mean(np.sum((probabilities - _one_hot(outcomes)) ** 2, axis=1)))


def log_loss(probabilities: Probabilities, outcomes: Outcomes) -> float:
    """Mean negative log probability of what happened.

    Unbounded above, which is the point: it punishes confident errors far harder
    than Brier does, and a model that is never surprised is usually overfitted.
    """
    _checked(probabilities, outcomes)
    hit = probabilities[np.arange(outcomes.shape[0]), outcomes]
    return float(-np.mean(np.log(np.clip(hit, FLOOR, 1.0))))


def ranked_probability_score(probabilities: Probabilities, outcomes: Outcomes) -> float:
    """Squared error between the cumulative distributions. Ranges 0 to 1.

    Home, draw and away are ordered, and this is the only one of the three that
    knows it. Calling an away win when the home side wins is a worse miss than
    calling a draw, and Brier scores those two identically.
    """
    _checked(probabilities, outcomes)
    predicted = np.cumsum(probabilities, axis=1)[:, :-1]
    actual = np.cumsum(_one_hot(outcomes), axis=1)[:, :-1]
    return float(np.mean(np.sum((predicted - actual) ** 2, axis=1)) / (CLASSES - 1))
