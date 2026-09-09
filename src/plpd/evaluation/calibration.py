"""Whether the probabilities mean what they say.

Brier, log loss and RPS each fold two different faults into one number. A model
can score well while being systematically wrong about how often its 30% calls
come in, and a well calibrated model can still be useless if it says the same
thing about every match. These pull the two apart.
"""

from dataclasses import dataclass

import numpy as np

from plpd.evaluation.metrics import CLASSES, Outcomes, Probabilities

BINS = 10


@dataclass(frozen=True)
class Bin:
    lower: float
    upper: float
    count: int
    predicted: float
    observed: float


@dataclass(frozen=True)
class Decomposition:
    """Murphy's three terms, summed over the three outcomes.

    Reliability is the penalty for saying 30% when it happens 40% of the time,
    and lower is better. Resolution rewards separating matches from the base
    rate, and higher is better. Uncertainty depends only on how the season
    turned out, so it is the same for every predictor and neither can be
    improved nor blamed on one.
    """

    reliability: float
    resolution: float
    uncertainty: float

    @property
    def brier(self) -> float:
        return self.reliability - self.resolution + self.uncertainty


def _happened(outcomes: Outcomes) -> Probabilities:
    encoded: Probabilities = (outcomes[:, None] == np.arange(CLASSES)).astype(np.float64)
    return encoded


def _bin_index(predicted: Probabilities, bins: int) -> Outcomes:
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed, so a forecast of exactly 1.0 lands in the top bin instead of
    # falling off the end of the table.
    return np.digitize(predicted, edges[1:-1], right=True).astype(np.int_)


def reliability_bins(
    probabilities: Probabilities, outcomes: Outcomes, *, bins: int = BINS
) -> list[Bin]:
    """Predicted against observed, pooled across all three outcomes.

    Pooled rather than split by outcome because 320 matches split three ways
    leaves bins too thin to read.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    predicted = probabilities.ravel()
    happened = _happened(outcomes).ravel()
    index = _bin_index(predicted, bins)
    edges = np.linspace(0.0, 1.0, bins + 1)

    table = []
    for slot in range(bins):
        inside = index == slot
        count = int(inside.sum())
        table.append(
            Bin(
                lower=float(edges[slot]),
                upper=float(edges[slot + 1]),
                count=count,
                predicted=float(predicted[inside].mean()) if count else 0.0,
                observed=float(happened[inside].mean()) if count else 0.0,
            )
        )
    return table


def decompose(
    probabilities: Probabilities, outcomes: Outcomes, *, bins: int = BINS
) -> Decomposition:
    """Split the Brier score into reliability, resolution and uncertainty.

    The identity is exact only when forecasts are grouped by identical value.
    Binning continuous ones makes it approximate, so `brier` reconstructs the
    scored Brier closely rather than exactly.
    """
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    happened = _happened(outcomes)
    total = len(outcomes)
    reliability = 0.0
    resolution = 0.0
    uncertainty = 0.0

    for outcome in range(CLASSES):
        predicted = probabilities[:, outcome]
        actual = happened[:, outcome]
        rate = float(actual.mean())
        uncertainty += rate * (1.0 - rate)

        index = _bin_index(predicted, bins)
        for slot in range(bins):
            inside = index == slot
            count = int(inside.sum())
            if not count:
                continue
            reliability += count * (predicted[inside].mean() - actual[inside].mean()) ** 2
            resolution += count * (actual[inside].mean() - rate) ** 2

    return Decomposition(reliability / total, resolution / total, uncertainty)


def outcome_rates(probabilities: Probabilities, outcomes: Outcomes) -> Probabilities:
    """Mean predicted probability against observed frequency, one row per outcome."""
    return np.column_stack([probabilities.mean(axis=0), _happened(outcomes).mean(axis=0)])
