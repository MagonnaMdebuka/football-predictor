"""Backtest dataclasses for count models (corners, cards)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CountPrediction:
    """Per-match count prediction record.

    Attributes:
        date: match date as ISO string
        season: season identifier
        home_team: home team name
        away_team: away team name
        model_type: 'corners' or 'cards'
        actual_home: actual count for home team
        actual_away: actual count for away team
        mu_home: predicted expected count for home team
        mu_away: predicted expected count for away team
        alpha: NB2 overdispersion parameter
        referee: referee name (cards only); None for corners
        match_over_under: list of {line, over, under} for total
        home_over_under: list of {line, over, under} for home team
        away_over_under: list of {line, over, under} for away team
        n_training_matches: matches in training set for this prediction
    """

    date: str
    season: str
    home_team: str
    away_team: str
    model_type: str
    actual_home: int
    actual_away: int
    mu_home: float
    mu_away: float
    alpha: float
    referee: str | None = None
    match_over_under: list[dict] = field(default_factory=list)
    home_over_under: list[dict] = field(default_factory=list)
    away_over_under: list[dict] = field(default_factory=list)
    n_training_matches: int = 0
    # Compound card model fields (None for corners and non-compound cards)
    mu_yellow_home: float | None = None
    mu_yellow_away: float | None = None
    alpha_yellow: float | None = None
    mu_red_home: float | None = None
    mu_red_away: float | None = None


@dataclass(frozen=True)
class CountMetricSet:
    """Evaluation metrics for count model predictions.

    Attributes:
        mean_brier: mean Brier score across all O/U lines
        per_line_brier: dict mapping line -> Brier score
        n_predictions: number of predictions evaluated
        mean_predicted_total: mean predicted total (mu_home + mu_away)
        mean_actual_total: mean actual total
    """

    mean_brier: float
    per_line_brier: dict[float, float]
    n_predictions: int
    mean_predicted_total: float
    mean_actual_total: float


@dataclass(frozen=True)
class CountCalibration:
    """Aggregate count calibration metrics.

    Attributes:
        n_predictions: number of predictions evaluated
        predicted_mean_total: mean of (mu_home + mu_away)
        actual_mean_total: mean of (actual_home + actual_away)
        bias: predicted - actual
    """

    n_predictions: int
    predicted_mean_total: float
    actual_mean_total: float
    bias: float


def compute_count_brier(
    predictions: list[CountPrediction],
) -> CountMetricSet:
    """Compute Brier scores for count model O/U predictions.

    For each O/U line, the Brier score is mean((p_over - indicator(actual > line))^2).
    The mean Brier is averaged across all lines.
    """
    if not predictions:
        return CountMetricSet(
            mean_brier=0.0, per_line_brier={}, n_predictions=0,
            mean_predicted_total=0.0, mean_actual_total=0.0,
        )

    # Gather all lines from the first prediction
    lines = [ou["line"] for ou in predictions[0].match_over_under]

    per_line_brier: dict[float, float] = {}

    for line in lines:
        brier_sum = 0.0
        count = 0
        for pred in predictions:
            actual_total = pred.actual_home + pred.actual_away
            # Find the O/U entry for this line
            for ou in pred.match_over_under:
                if ou["line"] == line:
                    indicator = 1.0 if actual_total > line else 0.0
                    brier_sum += (ou["over"] - indicator) ** 2
                    count += 1
                    break
        if count > 0:
            per_line_brier[line] = round(brier_sum / count, 6)

    mean_brier = (
        sum(per_line_brier.values()) / len(per_line_brier) if per_line_brier else 0.0
    )

    pred_totals = [p.mu_home + p.mu_away for p in predictions]
    actual_totals = [p.actual_home + p.actual_away for p in predictions]

    return CountMetricSet(
        mean_brier=round(mean_brier, 6),
        per_line_brier=per_line_brier,
        n_predictions=len(predictions),
        mean_predicted_total=round(sum(pred_totals) / len(pred_totals), 4),
        mean_actual_total=round(sum(actual_totals) / len(actual_totals), 4),
    )


def compute_count_calibration(
    predictions: list[CountPrediction],
) -> CountCalibration | None:
    """Compute calibration metrics for count predictions."""
    if not predictions:
        return None

    n = len(predictions)
    pred_mean = sum(p.mu_home + p.mu_away for p in predictions) / n
    actual_mean = sum(p.actual_home + p.actual_away for p in predictions) / n

    return CountCalibration(
        n_predictions=n,
        predicted_mean_total=round(pred_mean, 4),
        actual_mean_total=round(actual_mean, 4),
        bias=round(pred_mean - actual_mean, 4),
    )
