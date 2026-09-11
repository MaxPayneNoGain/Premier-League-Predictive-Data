from datetime import UTC, datetime, timedelta
from itertools import permutations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.db.tables import Prediction
from plpd.models import fit, predict
from plpd.predictions import MODEL, predict_round, record

SEASON_START = datetime(2025, 8, 16, 14, 0, tzinfo=UTC)

PREDICTED_AT = datetime(2026, 9, 10, 9, 0, tzinfo=UTC)

STRENGTH = {1: 3, 2: 2, 3: 2, 4: 1}


def stored(moment: datetime) -> datetime:
    # SQLite has no timestamp type and the offset does not survive the round trip.
    return moment.replace(tzinfo=None)


def history() -> pd.DataFrame:
    rows = []
    for index, (home, away) in enumerate(permutations(STRENGTH, 2)):
        rows.append(
            {
                "match_id": f"{home}-v-{away}",
                "season": "2025-2026",
                "gameweek": index + 1,
                "kickoff_time": SEASON_START + timedelta(days=7 * index),
                "home_team_code": home,
                "away_team_code": away,
                "home_goals": STRENGTH[home],
                "away_goals": STRENGTH[away] - 1,
            }
        )
    return pd.DataFrame(rows)


def round_of(*pairs: tuple[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": f"gw-{home}-v-{away}",
                "season": "2025-2026",
                "gameweek": 20,
                "kickoff_time": SEASON_START + timedelta(days=200),
                "home_team_code": home,
                "away_team_code": away,
            }
            for home, away in pairs
        ]
    )


def test_one_row_comes_back_per_fixture() -> None:
    predictions = predict_round(history(), round_of((1, 4), (2, 3)))

    assert list(predictions["match_id"]) == ["gw-1-v-4", "gw-2-v-3"]


def test_the_three_probabilities_sum_to_one() -> None:
    predictions = predict_round(history(), round_of((1, 4), (2, 3)))

    total = predictions[["home_win", "draw", "away_win"]].sum(axis=1)
    assert np.allclose(total, 1.0)


def test_the_stronger_side_at_home_is_favoured() -> None:
    predictions = predict_round(history(), round_of((1, 4))).iloc[0]

    assert predictions["home_win"] > predictions["away_win"]
    assert predictions["home_xg"] > predictions["away_xg"]


def test_predictions_are_shrunk_toward_the_base_rate() -> None:
    fixtures = round_of((1, 4))
    unshrunk = predict(fit(history(), correlation=True), fixtures)[0][0]

    assert predict_round(history(), fixtures).iloc[0]["home_win"] < unshrunk


def test_the_fitted_correlation_rides_along() -> None:
    predictions = predict_round(history(), round_of((1, 4), (2, 3)))

    assert predictions["rho"].nunique() == 1
    assert predictions["rho"].iloc[0] != 0.0


def test_a_round_with_no_history_cannot_be_predicted() -> None:
    with pytest.raises(ValueError):
        predict_round(history().iloc[:0], round_of((1, 4)))


def test_recording_writes_one_row_per_prediction(session: Session) -> None:
    predictions = predict_round(history(), round_of((1, 4), (2, 3)))

    written = record(
        session,
        predictions,
        predicted_at=PREDICTED_AT,
        trained_through=SEASON_START,
    )

    assert written == 2
    assert session.scalars(select(Prediction.match_id)).all() == ["gw-1-v-4", "gw-2-v-3"]


def test_a_recorded_row_carries_what_the_model_knew(session: Session) -> None:
    predictions = predict_round(history(), round_of((1, 4)))
    record(session, predictions, predicted_at=PREDICTED_AT, trained_through=SEASON_START)

    row = session.scalars(select(Prediction)).one()
    assert row.model == MODEL
    assert row.predicted_at == stored(PREDICTED_AT)
    assert row.trained_through == stored(SEASON_START)
    assert row.home_win == pytest.approx(predictions.iloc[0]["home_win"])


def test_a_second_run_is_kept_alongside_the_first(session: Session) -> None:
    predictions = predict_round(history(), round_of((1, 4)))
    record(session, predictions, predicted_at=PREDICTED_AT, trained_through=SEASON_START)
    record(
        session,
        predictions,
        predicted_at=PREDICTED_AT + timedelta(days=1),
        trained_through=SEASON_START,
    )

    assert len(session.scalars(select(Prediction)).all()) == 2


def test_nothing_to_record_writes_nothing(session: Session) -> None:
    empty = predict_round(history(), round_of((1, 4))).iloc[:0]

    assert record(session, empty, predicted_at=PREDICTED_AT, trained_through=SEASON_START) == 0
