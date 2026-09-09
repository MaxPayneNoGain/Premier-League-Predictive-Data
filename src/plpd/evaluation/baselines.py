"""Reference points a model has to beat to be worth anything.

A match model that cannot beat the league's own base rates has learned nothing
about the teams, only about football. Each of these is deliberately naive, and
each takes the same shape as a real prediction so the same scoring code covers
both.
"""

import numpy as np

from plpd.evaluation.metrics import CLASSES, Outcomes, Probabilities


def uniform(matches: int) -> Probabilities:
    """One third each. The floor: no information at all."""
    return np.full((matches, CLASSES), 1 / CLASSES)


def always_home(matches: int) -> Probabilities:
    """Certainty that the home side wins.

    Kept because it is the baseline people reach for first, and because what it
    does to log loss is the argument against reporting a prediction more
    confident than the evidence.
    """
    probabilities = np.zeros((matches, CLASSES))
    probabilities[:, 0] = 1.0
    return probabilities


def base_rates(trained_on: Outcomes, matches: int) -> Probabilities:
    """How often each result happened historically, applied to every match.

    Takes the outcomes it was fitted on rather than reading them itself, so a
    backtest cannot accidentally hand it the results it is about to be scored
    against.
    """
    if trained_on.size == 0:
        raise ValueError("base rates need at least one historical outcome")
    counts = np.bincount(trained_on, minlength=CLASSES)
    return np.tile(counts / counts.sum(), (matches, 1))
