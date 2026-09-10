import numpy as np
import pandas as pd
import pytest

from plpd.evaluation import devig, overround, prices, score_walk_forward

FAIR = np.array([[2.0, 4.0, 4.0]])

PRICED = np.array([[1.90, 3.60, 4.20], [2.50, 3.30, 2.80]])


def test_a_book_without_margin_is_left_alone() -> None:
    assert devig(FAIR) == pytest.approx(np.array([[0.5, 0.25, 0.25]]))


def test_de_vigged_prices_are_a_distribution() -> None:
    stripped = devig(PRICED)

    assert stripped.sum(axis=1) == pytest.approx([1.0, 1.0])
    assert (stripped > 0).all()


def test_the_shortest_price_stays_the_likeliest() -> None:
    stripped = devig(PRICED)

    assert stripped[0].argmax() == 0
    assert stripped[1].argmax() == 0


def test_de_vigging_lowers_every_implied_probability() -> None:
    assert (devig(PRICED) < 1.0 / PRICED).all()


def test_a_fair_book_has_no_overround() -> None:
    assert overround(FAIR) == pytest.approx([0.0])


def test_overround_is_the_bookmaker_margin() -> None:
    assert overround(PRICED) == pytest.approx([0.0422, 0.0602], abs=1e-4)


def test_odds_below_the_stake_are_refused() -> None:
    with pytest.raises(ValueError, match="must exceed 1"):
        devig(np.array([[1.0, 3.6, 4.2]]))


def test_a_partly_priced_match_is_refused() -> None:
    with pytest.raises(ValueError, match="only some outcomes"):
        devig(np.array([[1.9, np.nan, 4.2]]))


def test_two_way_odds_are_refused() -> None:
    with pytest.raises(ValueError, match="got"):
        devig(np.array([[1.9, 4.2]]))


def test_prices_come_off_the_frame_home_draw_away() -> None:
    frame = pd.DataFrame(
        {
            "match_id": ["a"],
            "away_odds": [4.2],
            "home_odds": [1.9],
            "draw_odds": [3.6],
        }
    )

    assert prices(frame) == pytest.approx(np.array([[1.9, 3.6, 4.2]]))


def test_a_frame_without_odds_is_refused() -> None:
    with pytest.raises(ValueError, match="draw_odds"):
        prices(pd.DataFrame({"match_id": ["a"], "home_odds": [1.9]}))


def priced_season(gameweeks: int = 8, per_week: int = 10) -> pd.DataFrame:
    start = pd.Timestamp("2025-08-09")
    return pd.DataFrame(
        [
            {
                "match_id": f"gw{week}-{game}",
                "season": "2025-2026",
                "gameweek": week,
                "kickoff_time": start + pd.Timedelta(weeks=week),
                "home_goals": 2,
                "away_goals": 1,
                "home_odds": 1.9,
                "draw_odds": 3.6,
                "away_odds": 4.2,
            }
            for week in range(1, gameweeks + 1)
            for game in range(per_week)
        ]
    )


def test_the_market_scores_as_a_predictor() -> None:
    scores = score_walk_forward(priced_season(), lambda _, test: devig(prices(test)))

    assert scores.matches == 20
    assert 0.0 < scores.rps < 1.0
