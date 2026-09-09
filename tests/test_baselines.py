import numpy as np
import pytest

from plpd.evaluation import always_home, base_rates, log_loss, uniform

HOME, DRAW, AWAY = 0, 1, 2


def test_uniform_splits_evenly() -> None:
    assert uniform(2).tolist() == [[1 / 3] * 3] * 2


def test_always_home_is_certain() -> None:
    assert always_home(2).tolist() == [[1.0, 0.0, 0.0]] * 2


def test_base_rates_repeat_the_training_frequencies() -> None:
    history = np.array([HOME, HOME, DRAW, AWAY])

    predicted = base_rates(history, matches=3)

    assert predicted.shape == (3, 3)
    assert predicted[0].tolist() == [0.5, 0.25, 0.25]


def test_base_rates_need_history() -> None:
    with pytest.raises(ValueError, match="at least one"):
        base_rates(np.array([], dtype=np.int_), matches=1)


def test_confidence_without_evidence_is_punished() -> None:
    # Two home wins and an away win. Always-home gets two right, the base rates
    # get none outright, and log loss still prefers the base rates.
    played = np.array([HOME, HOME, AWAY])

    assert log_loss(base_rates(played, 3), played) < log_loss(always_home(3), played)
