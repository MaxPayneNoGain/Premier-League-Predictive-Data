"""Database schema.

Raw pulls are archived as Parquet in full; only the columns a model actually
uses get promoted into a table here. More tables land as later phases need them.
"""

from datetime import datetime

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SEASON_LEN = 9
MATCH_ID_LEN = 128


class Base(DeclarativeBase):
    pass


class Snapshot(Base):
    """One archived pull of an upstream source."""

    __tablename__ = "snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    season: Mapped[str] = mapped_column(String(SEASON_LEN))

    # Upstream overwrites its files in place and its own news_added column comes
    # through empty, so this is the only trustworthy record of when a fact
    # became knowable.
    fetched_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    source_ref: Mapped[str] = mapped_column(String(64))
    storage_uri: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    row_count: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        # Backs the skip in archive(), and stops two concurrent runs both
        # storing the same pull.
        UniqueConstraint("source", "season", "content_hash", name="uq_snapshots_content"),
        Index("ix_snapshots_source_fetched_at", "source", "fetched_at"),
    )


class Team(Base):
    # code is stable across seasons, fpl_id is reassigned each season.
    __tablename__ = "teams"

    season: Mapped[str] = mapped_column(String(SEASON_LEN), primary_key=True)
    fpl_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(64))
    short_name: Mapped[str] = mapped_column(String(8))

    __table_args__ = (Index("ix_teams_code", "code"),)


class Player(Base):
    __tablename__ = "players"

    season: Mapped[str] = mapped_column(String(SEASON_LEN), primary_key=True)
    fpl_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[int] = mapped_column(Integer)
    first_name: Mapped[str] = mapped_column(String(64))
    second_name: Mapped[str] = mapped_column(String(64))
    web_name: Mapped[str] = mapped_column(String(64))
    team_code: Mapped[int] = mapped_column(Integer)
    position: Mapped[str] = mapped_column(String(16))

    __table_args__ = (Index("ix_players_code", "code"),)


class Odds(Base):
    """Bookmaker prices for one match, as decimal odds.

    Stored as odds rather than probabilities. Turning a price into a probability
    means deciding how to strip the bookmaker's margin, and that decision belongs
    where the probabilities are used.
    """

    __tablename__ = "odds"

    match_id: Mapped[str] = mapped_column(String(MATCH_ID_LEN), primary_key=True)
    bookmaker: Mapped[str] = mapped_column(String(32))
    home_win: Mapped[float] = mapped_column(Float)
    draw: Mapped[float] = mapped_column(Float)
    away_win: Mapped[float] = mapped_column(Float)


class Prediction(Base):
    """One model's prediction for one match. Insert only, never updated."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[str] = mapped_column(String(MATCH_ID_LEN))
    model: Mapped[str] = mapped_column(String(32))
    predicted_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    # The last kickoff in the training window.
    trained_through: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))

    home_win: Mapped[float] = mapped_column(Float)
    draw: Mapped[float] = mapped_column(Float)
    away_win: Mapped[float] = mapped_column(Float)
    home_xg: Mapped[float] = mapped_column(Float)
    away_xg: Mapped[float] = mapped_column(Float)
    rho: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("match_id", "model", "predicted_at", name="uq_predictions_run"),
        Index("ix_predictions_match_id", "match_id"),
    )


class Fixture(Base):
    """A scheduled or completed match in any competition.

    European ties live here too, because the rotation model needs a club's whole
    calendar to work out rest days. The team columns carry codes and not FPL
    ids, which is what upstream puts there and also the only key that survives a
    season rollover. They are nullable because upstream assigns them solely to
    clubs that exist in FPL, so a Barcelona row arrives with a name and nothing
    else.
    """

    __tablename__ = "fixtures"

    match_id: Mapped[str] = mapped_column(String(MATCH_ID_LEN), primary_key=True)
    season: Mapped[str] = mapped_column(String(SEASON_LEN))
    gameweek: Mapped[int] = mapped_column(Integer)
    tournament: Mapped[str] = mapped_column(String(32))
    kickoff_time: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    home_team_code: Mapped[int | None] = mapped_column(Integer)
    away_team_code: Mapped[int | None] = mapped_column(Integer)
    home_team_elo: Mapped[float | None] = mapped_column(Float)
    away_team_elo: Mapped[float | None] = mapped_column(Float)
    home_score: Mapped[int | None] = mapped_column(Integer)
    away_score: Mapped[int | None] = mapped_column(Integer)
    finished: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (Index("ix_fixtures_season_gameweek", "season", "gameweek"),)
