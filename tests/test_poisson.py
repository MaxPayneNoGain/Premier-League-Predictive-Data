import pandas as pd
import pytest

from plpd.models import expected_goals, fit, outcome_probabilities, predict, score_matrix
from plpd.models.poisson import RHO_BOUNDS, _decay_weights

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


SPREAD = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]


def draw_heavy(repeats: int = 2) -> pd.DataFrame:
    pairs = [(home, away) for home in sorted(SCORED) for away in sorted(SCORED) if home != away]
    rows = [
        (home, away, home_goals, away_goals)
        for _ in range(repeats)
        for home, away in pairs
        for home_goals, away_goals in SPREAD
    ]
    rows += [(home, away, goals, goals) for home, away in pairs for goals in (0, 1)]
    return league(rows)


def test_the_correction_leaves_higher_scorelines_in_proportion() -> None:
    plain = score_matrix(1.5, 1.2)
    corrected = score_matrix(1.5, 1.2, -0.1)

    assert corrected[2, 2] / corrected[3, 1] == pytest.approx(plain[2, 2] / plain[3, 1])


def test_a_negative_rho_lifts_the_low_draws_and_cuts_the_one_goal_wins() -> None:
    plain = score_matrix(1.5, 1.2)
    corrected = score_matrix(1.5, 1.2, -0.1)

    assert corrected[0, 0] > plain[0, 0]
    assert corrected[1, 1] > plain[1, 1]
    assert corrected[1, 0] < plain[1, 0]
    assert corrected[0, 1] < plain[0, 1]


def test_the_corrected_matrix_is_still_a_distribution() -> None:
    corrected = score_matrix(1.5, 1.2, -0.1)

    assert corrected.sum() == pytest.approx(1.0)
    assert (corrected >= 0).all()


def test_the_correction_is_off_unless_it_is_asked_for() -> None:
    model = fit(draw_heavy())

    assert model.rho == 0.0


def test_the_correction_finds_a_negative_rho_when_low_draws_pile_up() -> None:
    model = fit(draw_heavy(), correlation=True)

    assert model.rho < 0.0
    assert RHO_BOUNDS[0] < model.rho < RHO_BOUNDS[1]


def test_the_correction_raises_the_chance_of_a_draw() -> None:
    matches = draw_heavy()

    plain = predict(fit(matches), fixture(STRONG, WEAK))
    corrected = predict(fit(matches, correlation=True), fixture(STRONG, WEAK))

    assert corrected[0, DRAW] > plain[0, DRAW]


def reversal(repeats: int = 8) -> pd.DataFrame:
    early = [(STRONG, WEAK, 3, 1), (WEAK, STRONG, 1, 3)] * repeats
    late = [(STRONG, WEAK, 1, 3), (WEAK, STRONG, 3, 1)] * repeats
    frame = league(early + late)
    frame["kickoff_time"] = list(pd.date_range("2025-01-01", periods=len(early), freq="D")) + list(
        pd.date_range("2026-01-01", periods=len(late), freq="D")
    )
    return frame


def test_a_match_one_half_life_old_counts_half() -> None:
    matches = league([(STRONG, WEAK, 1, 0), (WEAK, STRONG, 1, 0)])
    matches["kickoff_time"] = [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")]

    weights = _decay_weights(matches, 90)

    assert weights[1] == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.5)


def test_without_decay_a_reversal_leaves_both_sides_level() -> None:
    model = fit(reversal())

    strong = model.attack[model.teams[STRONG]]
    weak = model.attack[model.teams[WEAK]]

    assert strong == pytest.approx(weak, abs=0.01)


def test_decay_lets_recent_form_outweigh_old_form() -> None:
    model = fit(reversal(), half_life=30)

    strong = model.attack[model.teams[STRONG]]
    weak = model.attack[model.teams[WEAK]]

    assert weak > strong + 0.5


def test_a_very_long_half_life_is_almost_no_decay() -> None:
    plain = fit(reversal())
    slow = fit(reversal(), half_life=100_000)

    assert slow.attack[slow.teams[WEAK]] == pytest.approx(plain.attack[plain.teams[WEAK]], abs=0.01)


def test_a_half_life_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        fit(reversal(), half_life=0)


def test_decay_needs_a_kickoff_time() -> None:
    with pytest.raises(ValueError, match="kickoff_time"):
        fit(synthetic(), half_life=30)
