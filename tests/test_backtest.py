import numpy as np
import pandas as pd
import pytest

from plpd.evaluation import score_walk_forward, walk_forward


def matches(*rows: tuple[str, int, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": match_id,
                "gameweek": gameweek,
                "kickoff_time": pd.Timestamp(kickoff),
                "home_goals": 2,
                "away_goals": 1,
            }
            for match_id, gameweek, kickoff in rows
        ]
    )


SEASON_START = pd.Timestamp("2025-08-09")


def season(gameweeks: int, per_week: int = 10) -> pd.DataFrame:
    return matches(
        *(
            (f"gw{week}-{game}", week, str(SEASON_START + pd.Timedelta(weeks=week)))
            for week in range(1, gameweeks + 1)
            for game in range(per_week)
        )
    )


def test_folds_cut_on_time_not_gameweek() -> None:
    played = matches(
        ("gw1-played", 1, "2025-08-16"),
        ("gw1-postponed", 1, "2025-09-30"),
        ("gw2", 2, "2025-08-23"),
    )

    fold = next(f for f in walk_forward(played, min_train=1) if f.gameweek == 2)

    assert list(fold.train["match_id"]) == ["gw1-played"]


def test_no_fold_trains_on_a_match_it_is_about_to_predict() -> None:
    for fold in walk_forward(season(12), min_train=10):
        assert set(fold.train["match_id"]).isdisjoint(fold.test["match_id"])


def test_every_training_match_kicked_off_before_every_test_match() -> None:
    for fold in walk_forward(season(12), min_train=10):
        assert fold.train["kickoff_time"].max() < fold.test["kickoff_time"].min()


def test_gameweeks_without_enough_history_are_skipped() -> None:
    folds = list(walk_forward(season(10), min_train=60))

    assert [fold.gameweek for fold in folds] == [7, 8, 9, 10]


def test_folds_arrive_oldest_first() -> None:
    weeks = [fold.gameweek for fold in walk_forward(season(12), min_train=10)]

    assert weeks == sorted(weeks)


def test_a_nonsense_minimum_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        list(walk_forward(season(4), min_train=0))


def always_right(_: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.tile([1.0, 0.0, 0.0], (len(test), 1))


def test_a_predictor_that_is_always_right_scores_zero() -> None:
    scores = score_walk_forward(season(12), always_right, min_train=10)

    assert (scores.brier, scores.log_loss, scores.rps) == (0.0, 0.0, 0.0)


def test_every_predicted_match_is_counted_once() -> None:
    played = season(12)

    scores = score_walk_forward(played, always_right, min_train=10)
    folds = list(walk_forward(played, min_train=10))

    assert scores.matches == sum(len(fold.test) for fold in folds)


def test_a_season_too_short_to_learn_from_is_rejected() -> None:
    with pytest.raises(ValueError, match="earlier matches"):
        score_walk_forward(season(2), always_right, min_train=60)
