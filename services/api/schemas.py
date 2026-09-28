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
