"""Bookmaker prices as the baseline the model is measured against.

Decimal odds carry the bookmaker's margin, so their reciprocals sum to more than
one. Dividing through by that sum spreads the margin across the three prices in
proportion to each. Real books load more of the margin onto the longer prices,
and correcting for that needs a model of the bias. This does not attempt one.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from plpd.evaluation.metrics import CLASSES, Probabilities
from plpd.features import PRICE_COLUMNS

Prices = npt.NDArray[np.float64]


def devig(odds: Prices) -> Probabilities:
    """Strip the margin from decimal odds, leaving three probabilities."""
    implied = _implied(odds)
    stripped: Probabilities = implied / implied.sum(axis=1, keepdims=True)
    return stripped


def overround(odds: Prices) -> Prices:
    """How far above 1 the implied probabilities sum, one figure per match."""
    excess: Prices = _implied(odds).sum(axis=1) - 1.0
    return excess


def prices(matches: pd.DataFrame) -> Prices:
    """Read the three decimal prices off a matches frame, home, draw, away."""
    missing = [column for column in PRICE_COLUMNS if column not in matches]
    if missing:
        raise ValueError(f"the frame carries no odds, missing {', '.join(missing)}")
    read: Prices = matches.loc[:, PRICE_COLUMNS].to_numpy(dtype=np.float64)
    return read


def _implied(odds: Prices) -> Prices:
    if odds.ndim != 2 or odds.shape[1] != CLASSES:
        raise ValueError(f"expected an (n, {CLASSES}) array of odds, got {odds.shape}")
    if not np.isfinite(odds).all():
        raise ValueError("a match priced for only some outcomes cannot be de-vigged")
    if (odds <= 1.0).any():
        raise ValueError("decimal odds return the stake as well, so they must exceed 1")
    reciprocal: Prices = 1.0 / odds
    return reciprocal
