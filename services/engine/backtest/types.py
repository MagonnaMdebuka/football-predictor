"""Backtest dataclasses — all frozen for immutability and hashability."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BookmakerOddsCols:
    """Column names for bookmaker odds in raw JSON.

    Primary: Pinnacle closing odds.
    Fallback: market average closing odds.
    """

    primary_home: str = "PSCH"
    primary_draw: str = "PSCD"
    primary_away: str = "PSCA"
    fallback_home: str = "AvgCH"
    fallback_draw: str = "AvgCD"
    fallback_away: str = "AvgCA"


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a walk-forward backtest run.

    Attributes:
        held_out_seasons: seasons used for evaluation (not training)
        training_start_season: earliest season included in training data
        xi: time-decay rate for Dixon-Coles weights
        refit_step: 'per_date' refits on every distinct match date;
            'weekly' buckets by the most recent Monday on or before the prediction date
        weekly_refit_day: ISO weekday for weekly bucketing (0=Monday)
        rho_bounds: bounds for the Dixon-Coles rho parameter
        max_goals: grid size for score predictions
        seed: random seed for reproducibility
        bookmaker_odds_cols: column names for extracting bookmaker odds
        league_code: league code for capability flags (None = goals only)
        corners_xi: time-decay rate for corners model (None = use goals xi)
        cards_xi: time-decay rate for cards model (None = use goals xi)
        min_referee_matches: minimum matches for referee effect in cards model
    """

    held_out_seasons: tuple[str, ...]
    training_start_season: str
    xi: float = 0.0065
    refit_step: str = "per_date"
    weekly_refit_day: int = 0
    rho_bounds: tuple[float, float] = (-0.5, 0.5)
    max_goals: int = 11
    seed: int = 42
    bookmaker_odds_cols: BookmakerOddsCols = field(default_factory=BookmakerOddsCols)
    league_code: str | None = None
    corners_xi: float | None = None
    cards_xi: float | None = None
    min_referee_matches: int = 20


@dataclass(frozen=True)
class MetricSet:
    """Evaluation metrics for a set of predictions.

    Attributes:
        rps: ranked probability score (lower is better)
        log_loss: categorical log loss (lower is better)
        brier_home: Brier score for home-win probability
        brier_draw: Brier score for draw probability
        brier_away: Brier score for away-win probability
        n_matches: number of matches evaluated
        hit_rate: fraction of correct most-likely outcome predictions
    """

    rps: float
    log_loss: float
    brier_home: float
    brier_draw: float
    brier_away: float
    n_matches: int
    hit_rate: float


@dataclass(frozen=True)
class MatchPrediction:
    """Per-match prediction record with model and baseline probabilities.

    Attributes:
        date: match date as ISO string
        season: season identifier (e.g. '2024-25')
        home_team: home team name
        away_team: away team name
        home_goals: actual home goals scored
        away_goals: actual away goals scored
        result: actual result ('H', 'D', 'A')
        matchday: matchday number within the season
        model_home: model P(home win)
        model_draw: model P(draw)
        model_away: model P(away win)
        uniform_home: uniform baseline P(home win) = 1/3
        uniform_draw: uniform baseline P(draw) = 1/3
        uniform_away: uniform baseline P(away win) = 1/3
        base_rate_home: base-rate P(home win) from training data
        base_rate_draw: base-rate P(draw) from training data
        base_rate_away: base-rate P(away win) from training data
        indep_poisson_home: independent Poisson P(home win)
        indep_poisson_draw: independent Poisson P(draw)
        indep_poisson_away: independent Poisson P(away win)
        bookmaker_home: bookmaker P(home win), None if odds missing
        bookmaker_draw: bookmaker P(draw), None if odds missing
        bookmaker_away: bookmaker P(away win), None if odds missing
        n_training_matches: matches in training set for this prediction
    """

    date: str
    season: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    result: str
    matchday: int
    model_home: float
    model_draw: float
    model_away: float
    uniform_home: float
    uniform_draw: float
    uniform_away: float
    base_rate_home: float
    base_rate_draw: float
    base_rate_away: float
    indep_poisson_home: float
    indep_poisson_draw: float
    indep_poisson_away: float
    bookmaker_home: float | None
    bookmaker_draw: float | None
    bookmaker_away: float | None
    ablation_home: float = 0.0
    ablation_draw: float = 0.0
    ablation_away: float = 0.0
    lambda_home: float = 0.0
    lambda_away: float = 0.0
    n_training_matches: int = 0
    fallback_teams: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SeasonMetrics:
    """Per-season metric breakdown across all baselines.

    Attributes:
        season: season identifier
        model: model metrics
        uniform: uniform baseline metrics
        base_rate: base-rate baseline metrics
        independent_poisson: independent Poisson baseline metrics
        bookmaker: bookmaker baseline metrics (None if no odds available)
        bookmaker_exclusion_count: matches excluded from bookmaker comparison
        early_season: metrics for first 6 matchdays only (None if not computed)
    """

    season: str
    model: MetricSet
    uniform: MetricSet
    base_rate: MetricSet
    independent_poisson: MetricSet
    ablation: MetricSet | None
    bookmaker: MetricSet | None
    bookmaker_exclusion_count: int
    early_season: MetricSet | None


@dataclass(frozen=True)
class GateDetail:
    """Single gate check result."""

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class GoalCalibration:
    """Aggregate goal-calibration metrics across held-out matches.

    Compares predicted goal rates to actual rates to detect systematic
    bias that would propagate to every total-based market.

    Attributes:
        n_matches: number of matches evaluated
        predicted_mean_total: mean of (lambda_home + lambda_away)
        actual_mean_total: mean of (home_goals + away_goals)
        predicted_over_25_rate: mean of model P(over 2.5)
        actual_over_25_rate: fraction of matches with total goals > 2.5
    """

    n_matches: int
    predicted_mean_total: float
    actual_mean_total: float
    predicted_over_25_rate: float
    actual_over_25_rate: float


@dataclass(frozen=True)
class CountMetricSummary:
    """Serialisable summary of count model metrics and gate results.

    Attributes:
        mean_brier: mean Brier score across all O/U lines
        per_line_brier: dict mapping line -> Brier score
        n_predictions: number of predictions evaluated
        mean_predicted_total: mean predicted total
        mean_actual_total: mean actual total
    """

    mean_brier: float
    per_line_brier: dict[float, float]
    n_predictions: int
    mean_predicted_total: float
    mean_actual_total: float


@dataclass(frozen=True)
class CountCalibrationSummary:
    """Serialisable count calibration summary.

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


@dataclass(frozen=True)
class BacktestReport:
    """Full backtest report with all results and metadata.

    Attributes:
        schema_version: report format version
        created_at: ISO timestamp of report creation
        git_commit: short git hash at time of run
        config: backtest configuration used
        seasons: per-season metric breakdowns
        combined: combined metrics across all held-out seasons
        early_season_combined: combined metrics for first 6 matchdays
        predictions: per-match prediction records
        gate_passed: whether all gate checks passed
        gate_details: individual gate check results
        goal_calibration: aggregate goal-scoring calibration check
        corner_predictions: per-match corner predictions
        card_predictions: per-match card predictions
        corner_metrics: corner model evaluation metrics
        card_metrics: card model evaluation metrics
        corner_calibration: corner calibration summary
        card_calibration: card calibration summary
        corner_gate_passed: whether corner gate checks passed
        card_gate_passed: whether card gate checks passed
        corner_gate_details: corner gate check results
        card_gate_details: card gate check results
    """

    schema_version: str
    created_at: str
    git_commit: str
    config: BacktestConfig
    seasons: list[SeasonMetrics]
    combined: SeasonMetrics
    early_season_combined: SeasonMetrics | None
    predictions: list[MatchPrediction]
    gate_passed: bool
    gate_details: list[GateDetail]
    baseline_configs: dict = field(default_factory=dict)
    goal_calibration: GoalCalibration | None = None
    corner_predictions: list = field(default_factory=list)
    card_predictions: list = field(default_factory=list)
    corner_metrics: CountMetricSummary | None = None
    card_metrics: CountMetricSummary | None = None
    corner_calibration: CountCalibrationSummary | None = None
    card_calibration: CountCalibrationSummary | None = None
    corner_gate_passed: bool | None = None
    card_gate_passed: bool | None = None
    corner_gate_details: list[GateDetail] = field(default_factory=list)
    card_gate_details: list[GateDetail] = field(default_factory=list)
