import numpy as np
import pytest

from plpd.evaluation import brier_score, decompose, outcome_rates, reliability_bins

HOME, DRAW, AWAY = 0, 1, 2

SPREAD = [
    [0.60, 0.25, 0.15],
    [0.45, 0.30, 0.25],
    [0.30, 0.30, 0.40],
    [0.20, 0.25, 0.55],
]

# No two different forecasts for the same outcome share a bin, which is the
# condition under which the decomposition is exact rather than approximate.
UNSHARED = [
    [0.65, 0.15, 0.20],
    [0.45, 0.35, 0.20],
    [0.25, 0.55, 0.20],
]

PATTERN = [HOME, DRAW, AWAY, HOME, AWAY]


def flat(matches: int) -> np.ndarray:
    return np.full((matches, 3), 1 / 3)


def varied(matches: int) -> np.ndarray:
    return np.array([SPREAD[row % len(SPREAD)] for row in range(matches)])


def unshared(matches: int) -> np.ndarray:
    return np.array([UNSHARED[row % len(UNSHARED)] for row in range(matches)])


def thirds(matches: int) -> np.ndarray:
    return (np.arange(matches) % 3).astype(np.int_)


def mixed(matches: int) -> np.ndarray:
    return np.array([PATTERN[row % len(PATTERN)] for row in range(matches)], dtype=np.int_)


def test_a_calibrated_forecast_carries_almost_no_reliability_penalty() -> None:
    parts = decompose(flat(300), thirds(300))

    assert parts.reliability == pytest.approx(0.0, abs=1e-9)


def test_a_confident_forecast_that_is_wrong_is_penalised() -> None:
    parts = decompose(np.tile([0.9, 0.05, 0.05], (300, 1)), thirds(300))

    assert parts.reliability > 0.3


def test_resolution_is_zero_when_every_match_gets_the_same_forecast() -> None:
    parts = decompose(flat(300), thirds(300))

    assert parts.resolution == pytest.approx(0.0, abs=1e-9)


def test_uncertainty_depends_only_on_the_results() -> None:
    outcomes = thirds(300)

    even = decompose(flat(300), outcomes)
    confident = decompose(np.tile([0.9, 0.05, 0.05], (300, 1)), outcomes)

    assert even.uncertainty == pytest.approx(confident.uncertainty)


def test_the_three_terms_reconstruct_the_brier_score_exactly() -> None:
    probabilities = unshared(300)
    outcomes = mixed(300)

    parts = decompose(probabilities, outcomes)

    assert parts.brier == pytest.approx(brier_score(probabilities, outcomes))


def test_binning_forecasts_together_costs_only_a_little_accuracy() -> None:
    probabilities = varied(320)
    outcomes = thirds(320)

    parts = decompose(probabilities, outcomes)

    assert parts.brier == pytest.approx(brier_score(probabilities, outcomes), abs=0.01)


def test_every_forecast_lands_in_exactly_one_bin() -> None:
    probabilities = varied(320)

    table = reliability_bins(probabilities, thirds(320))

    assert sum(slot.count for slot in table) == probabilities.size


def test_binning_needs_at_least_one_bin() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        reliability_bins(flat(3), thirds(3), bins=0)


def test_outcome_rates_report_predicted_against_observed() -> None:
    rates = outcome_rates(flat(300), thirds(300))

    assert rates.shape == (3, 2)
    assert rates[HOME, 0] == pytest.approx(1 / 3)
    assert rates[AWAY, 1] == pytest.approx(1 / 3)


def test_outcome_rates_expose_an_under_predicted_outcome() -> None:
    rates = outcome_rates(np.tile([0.5, 0.2, 0.3], (300, 1)), thirds(300))

    assert rates[DRAW, 0] < rates[DRAW, 1]
