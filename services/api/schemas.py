"""Pydantic v2 response schemas for the Football Predictor API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class LeagueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    country: str
    code: str
    is_active: bool
    ship_corners: bool
    ship_cards: bool


class PredictionSummary(BaseModel):
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    home_expected_goals: float | None = None
    away_expected_goals: float | None = None
    most_likely_score: str | None = None
    confidence: str | None = None
    scoreline_note: str | None = None


class FixtureOut(BaseModel):
    id: int
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: str
    league_code: str | None = None
    league_name: str | None = None
    ft_home_goals: int | None = None
    ft_away_goals: int | None = None
    prediction: PredictionSummary | None = None


class MarketOut(BaseModel):
    market: str
    selection: str
    probability: float
    fair_odds: float | None = None
    line: float | None = None


class RefereeStatsOut(BaseModel):
    name: str
    matches_officiated: int
    avg_cards_per_match: float
    avg_yellows_per_match: float
    avg_reds_per_match: float


class MatchDetailOut(BaseModel):
    id: int
    home_team: str
    away_team: str
    kickoff_utc: datetime
    status: str
    ft_home_goals: int | None = None
    ft_away_goals: int | None = None
    referee: RefereeStatsOut | None = None
    prediction: PredictionSummary | None = None
    markets: dict[str, list[MarketOut]] = {}
    grid: list[list[float]] | None = None


class StandingsRowOut(BaseModel):
    team: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int


class LeagueDetailOut(BaseModel):
    league: LeagueOut
    fixtures: list[FixtureOut]
    standings: list[StandingsRowOut]


# --- Accuracy / Calibration schemas ---


class CalibrationBinOut(BaseModel):
    bin_lower: float
    bin_upper: float
    predicted_frequency: float
    observed_frequency: float
    sample_size: int


class ReliabilityDiagramOut(BaseModel):
    bins: list[CalibrationBinOut]
    calibration_error: float
    mean_calibration_error: float


class QualityBadgeOut(BaseModel):
    market: str
    badge: str
    n_seasons: int
    has_direct_data: bool


class GateStatusOut(BaseModel):
    name: str
    passed: bool
    message: str


class MarketAccuracyOut(BaseModel):
    market: str
    brier: float | None = None
    rps: float | None = None
    log_loss: float | None = None
    hit_rate: float | None = None
    sample_size: int
    badge: QualityBadgeOut
    gates: list[GateStatusOut]
    source: str


class AccuracyOverviewOut(BaseModel):
    markets: list[MarketAccuracyOut]
    total_settled: int
    source: str


class MarketAccuracyDetailOut(BaseModel):
    market: str
    reliability: ReliabilityDiagramOut
    calibration_bins: list[CalibrationBinOut]
    brier: float | None = None
    rps: float | None = None
    log_loss: float | None = None
    hit_rate: float | None = None
    sample_size: int
    badge: QualityBadgeOut
    gates: list[GateStatusOut]
    source: str


class ModelVsMarketItemOut(BaseModel):
    market: str
    model_metric: float
    bookmaker_metric: float
    metric_name: str
    sample_size: int


class ModelVsMarketOut(BaseModel):
    comparisons: list[ModelVsMarketItemOut]
    source: str
