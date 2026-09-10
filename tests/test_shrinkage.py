import numpy as np
import pytest

from plpd.models import shrink

CONFIDENT = np.array([[0.80, 0.12, 0.08], [0.05, 0.15, 0.80]])
REFERENCE = np.array([[0.44, 0.26, 0.30], [0.44, 0.26, 0.30]])


def test_no_weight_leaves_the_model_alone() -> None:
    assert shrink(CONFIDENT, REFERENCE, 0.0) == pytest.approx(CONFIDENT)


def test_full_weight_discards_the_model() -> None:
    assert shrink(CONFIDENT, REFERENCE, 1.0) == pytest.approx(REFERENCE)


def test_shrinking_still_leaves_a_distribution() -> None:
    blended = shrink(CONFIDENT, REFERENCE, 0.2)

    assert blended.sum(axis=1) == pytest.approx(1.0)
    assert (blended >= 0).all()


def test_confident_calls_are_pulled_toward_the_reference() -> None:
    blended = shrink(CONFIDENT, REFERENCE, 0.2)

    assert blended[0, 0] < CONFIDENT[0, 0]
    assert blended[0, 0] > REFERENCE[0, 0]
    assert blended[0, 2] > CONFIDENT[0, 2]


def test_the_ordering_of_a_row_survives() -> None:
    blended = shrink(CONFIDENT, REFERENCE, 0.2)

    assert blended[0].argmax() == CONFIDENT[0].argmax()
    assert blended[1].argmax() == CONFIDENT[1].argmax()


def test_a_weight_outside_the_unit_interval_is_rejected() -> None:
    for weight in (-0.1, 1.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            shrink(CONFIDENT, REFERENCE, weight)
