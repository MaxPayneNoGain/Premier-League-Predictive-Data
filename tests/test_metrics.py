import math

import numpy as np
import pytest

from plpd.evaluation import brier_score, log_loss, ranked_probability_score

CERTAIN_AND_RIGHT = np.array([[1.0, 0.0, 0.0]])
CERTAIN_AND_WRONG = np.array([[0.0, 0.0, 1.0]])
UNIFORM = np.array([[1 / 3, 1 / 3, 1 / 3]])
HOME_WON = np.array([0])


def test_a_certain_correct_call_scores_zero() -> None:
    assert brier_score(CERTAIN_AND_RIGHT, HOME_WON) == 0.0
    assert log_loss(CERTAIN_AND_RIGHT, HOME_WON) == 0.0
    assert ranked_probability_score(CERTAIN_AND_RIGHT, HOME_WON) == 0.0


def test_a_certain_wrong_call_scores_the_maximum() -> None:
    assert brier_score(CERTAIN_AND_WRONG, HOME_WON) == 2.0
    assert ranked_probability_score(CERTAIN_AND_WRONG, HOME_WON) == 1.0


def test_the_uniform_guess_matches_the_hand_computed_values() -> None:
    assert brier_score(UNIFORM, HOME_WON) == pytest.approx(2 / 3)
    assert log_loss(UNIFORM, HOME_WON) == pytest.approx(math.log(3))
    assert ranked_probability_score(UNIFORM, HOME_WON) == pytest.approx(5 / 18)


def test_only_rps_notices_that_the_outcomes_are_ordered() -> None:
    # Home won. Calling a draw is a near miss; calling an away win is not.
    near_miss = np.array([[0.0, 1.0, 0.0]])

    assert brier_score(near_miss, HOME_WON) == brier_score(CERTAIN_AND_WRONG, HOME_WON)
    assert ranked_probability_score(near_miss, HOME_WON) == 0.5
    assert ranked_probability_score(CERTAIN_AND_WRONG, HOME_WON) == 1.0


def test_log_loss_of_an_impossible_outcome_stays_finite() -> None:
    assert math.isfinite(log_loss(CERTAIN_AND_WRONG, HOME_WON))


def test_probabilities_that_do_not_sum_to_one_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        brier_score(np.array([[0.5, 0.2, 0.1]]), HOME_WON)


def test_a_length_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="against"):
        brier_score(UNIFORM, np.array([0, 1]))


def test_a_wrongly_shaped_array_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        brier_score(np.array([1.0, 0.0, 0.0]), HOME_WON)
