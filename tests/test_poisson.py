import pandas as pd
import pytest

from plpd.models import expected_goals, fit, outcome_probabilities, predict, score_matrix

HOME, DRAW, AWAY = 0, 1, 2

STRONG, WEAK, MIDDLE, OTHER = 3, 6, 9, 12
SCORED = {STRONG: 3, WEAK: 1, MIDDLE: 2, OTHER: 2}


def league(rows: list[tuple[int, int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["home_team_code", "away_team_code", "home_goals", "away_goals"]
    )


def synthetic(repeats: int = 6, home_bonus: int = 0) -> pd.DataFrame:
    codes = sorted(SCORED)
    return league(
        [
            (home, away, SCORED[home] + home_bonus, SCORED[away])
            for _ in range(repeats)
            for home in codes
            for away in codes
            if home != away
        ]
    )


def fixture(home: int, away: int) -> pd.DataFrame:
    return pd.DataFrame([{"home_team_code": home, "away_team_code": away}])


def test_a_score_matrix_is_a_distribution() -> None:
    matrix = score_matrix(1.6, 1.2)

    assert matrix.shape == (11, 11)
    assert matrix.sum() == pytest.approx(1.0)


def test_outcomes_come_out_of_the_matrix_summing_to_one() -> None:
    probabilities = outcome_probabilities(score_matrix(1.6, 1.2))

    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[HOME] > probabilities[AWAY]


def test_an_even_match_is_symmetric() -> None:
    probabilities = outcome_probabilities(score_matrix(1.4, 1.4))

    assert probabilities[HOME] == pytest.approx(probabilities[AWAY])


def test_the_side_that_scores_more_gets_the_higher_attack() -> None:
    model = fit(synthetic())

    assert model.attack[model.teams[STRONG]] > model.attack[model.teams[MIDDLE]]
    assert model.attack[model.teams[MIDDLE]] > model.attack[model.teams[WEAK]]


def test_strengths_are_pinned_to_sum_to_zero() -> None:
    model = fit(synthetic())

    assert model.attack.sum() == pytest.approx(0.0)
    assert model.defence.sum() == pytest.approx(0.0)


def test_home_advantage_is_zero_without_a_home_bias() -> None:
    model = fit(synthetic())

    assert model.home_advantage == pytest.approx(0.0, abs=0.01)


def test_home_advantage_is_found_when_it_is_there() -> None:
    model = fit(synthetic(home_bonus=1))

    assert model.home_advantage > 0.2


def test_expected_goals_favour_the_stronger_side() -> None:
    model = fit(synthetic())

    home_rate, away_rate = expected_goals(model, STRONG, WEAK)

    assert home_rate > away_rate


def test_predictions_are_one_row_per_fixture() -> None:
    model = fit(synthetic())

    probabilities = predict(model, fixture(STRONG, WEAK))

    assert probabilities.shape == (1, 3)
    assert probabilities.sum(axis=1) == pytest.approx(1.0)
    assert probabilities[0].argmax() == HOME


def test_a_club_never_seen_in_training_is_treated_as_average() -> None:
    model = fit(synthetic())

    probabilities = predict(model, fixture(STRONG, 99))

    assert probabilities.sum() == pytest.approx(1.0)


def test_fitting_needs_matches() -> None:
    with pytest.raises(ValueError, match="at least one match"):
        fit(league([]))
