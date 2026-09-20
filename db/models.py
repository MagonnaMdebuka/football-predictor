"""SQLAlchemy ORM models for the Football Predictor database.

15 tables covering leagues, teams, matches, predictions, and calibration.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class League(Base):
    __tablename__ = "leagues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    tier: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    seasons: Mapped[list["Season"]] = relationship(back_populates="league")
    matches: Mapped[list["Match"]] = relationship(back_populates="league")


class Season(Base):
    __tablename__ = "seasons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False)
    label: Mapped[str] = mapped_column(String(20), nullable=False)
    start_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    league: Mapped["League"] = relationship(back_populates="seasons")

    __table_args__ = (UniqueConstraint("league_id", "label", name="uq_season_league_label"),)


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    country: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    aliases: Mapped[list["TeamAlias"]] = relationship(back_populates="team")


class TeamAlias(Base):
    __tablename__ = "team_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_name: Mapped[str] = mapped_column(String(200), nullable=False)

    team: Mapped["Team"] = relationship(back_populates="aliases")

    __table_args__ = (UniqueConstraint("source", "raw_name", name="uq_alias_source_name"),)


class Referee(Base):
    __tablename__ = "referees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    country: Mapped[str | None] = mapped_column(String(100))


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id"), nullable=False)
    season_id: Mapped[int | None] = mapped_column(ForeignKey("seasons.id"))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    referee_id: Mapped[int | None] = mapped_column(ForeignKey("referees.id"))
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default="scheduled")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    league: Mapped["League"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(foreign_keys=[home_team_id])
    away_team: Mapped["Team"] = relationship(foreign_keys=[away_team_id])

    __table_args__ = (
        UniqueConstraint(
            "league_id",
            "kickoff_utc",
            "home_team_id",
            "away_team_id",
            name="uq_match_natural_key",
        ),
    )


class EloRating(Base):
    __tablename__ = "elo_ratings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    rating_delta: Mapped[float] = mapped_column(Float, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    parameters: Mapped[str | None] = mapped_column(Text)
    training_window_start: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    training_window_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    strengths: Mapped[list["TeamStrength"]] = relationship(back_populates="model_run")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="model_run")


class TeamStrength(Base):
    __tablename__ = "team_strengths"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    attack: Mapped[float] = mapped_column(Float, nullable=False)
    defence: Mapped[float] = mapped_column(Float, nullable=False)
    home_advantage: Mapped[float | None] = mapped_column(Float)

    model_run: Mapped["ModelRun"] = relationship(back_populates="strengths")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    home_win_prob: Mapped[float] = mapped_column(Float, nullable=False)
    draw_prob: Mapped[float] = mapped_column(Float, nullable=False)
    away_win_prob: Mapped[float] = mapped_column(Float, nullable=False)
    home_expected_goals: Mapped[float | None] = mapped_column(Float)
    away_expected_goals: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    model_run: Mapped["ModelRun"] = relationship(back_populates="predictions")
    market_predictions: Mapped[list["MarketPrediction"]] = relationship(
        back_populates="prediction"
    )


class MarketPrediction(Base):
    __tablename__ = "market_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), nullable=False)
    market: Mapped[str] = mapped_column(String(50), nullable=False)
    selection: Mapped[str] = mapped_column(String(50), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    line: Mapped[float | None] = mapped_column(Float)

    prediction: Mapped["Prediction"] = relationship(back_populates="market_predictions")


class CalibrationMap(Base):
    __tablename__ = "calibration_maps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    market: Mapped[str] = mapped_column(String(50), nullable=False)
    bin_lower: Mapped[float] = mapped_column(Float, nullable=False)
    bin_upper: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_frequency: Mapped[float] = mapped_column(Float, nullable=False)
    observed_frequency: Mapped[float] = mapped_column(Float, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)


class Outcome(Base):
    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False, unique=True)
    result: Mapped[str] = mapped_column(String(10), nullable=False)
    home_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    away_goals: Mapped[int] = mapped_column(Integer, nullable=False)
    settled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AccuracyMetric(Base):
    __tablename__ = "accuracy_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    sample_size: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="running")
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
