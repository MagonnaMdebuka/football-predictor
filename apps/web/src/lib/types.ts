/** TypeScript types mirroring the API Pydantic schemas. */

export interface LeagueOut {
  id: number;
  name: string;
  country: string;
  code: string;
  is_active: boolean;
  ship_corners: boolean;
  ship_cards: boolean;
}

export interface PredictionSummary {
  home_win_prob: number;
  draw_prob: number;
  away_win_prob: number;
  home_expected_goals: number | null;
  away_expected_goals: number | null;
  most_likely_score: string | null;
  confidence: string | null;
  scoreline_note: string | null;
}

export interface FixtureOut {
  id: number;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  status: string;
  league_code: string | null;
  league_name: string | null;
  ft_home_goals: number | null;
  ft_away_goals: number | null;
  prediction: PredictionSummary | null;
}

export interface MarketOut {
  market: string;
  selection: string;
  probability: number;
  fair_odds: number | null;
  line: number | null;
}

export interface RefereeStatsOut {
  name: string;
  matches_officiated: number;
  avg_cards_per_match: number;
  avg_yellows_per_match: number;
  avg_reds_per_match: number;
}

export interface MatchDetailOut {
  id: number;
  home_team: string;
  away_team: string;
  kickoff_utc: string;
  status: string;
  ft_home_goals: number | null;
  ft_away_goals: number | null;
  referee: RefereeStatsOut | null;
  prediction: PredictionSummary | null;
  markets: Record<string, MarketOut[]>;
  grid: number[][] | null;
}

export interface StandingsRowOut {
  team: string;
  played: number;
  won: number;
  drawn: number;
  lost: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
  points: number;
}

export interface LeagueDetailOut {
  league: LeagueOut;
  fixtures: FixtureOut[];
  standings: StandingsRowOut[];
}

// --- Accuracy / Calibration types ---

export interface CalibrationBinOut {
  bin_lower: number;
  bin_upper: number;
  predicted_frequency: number;
  observed_frequency: number;
  sample_size: number;
}

export interface ReliabilityDiagramOut {
  bins: CalibrationBinOut[];
  calibration_error: number;
  mean_calibration_error: number;
}

export interface QualityBadgeOut {
  market: string;
  badge: string;
  n_seasons: number;
  has_direct_data: boolean;
}

export interface GateStatusOut {
  name: string;
  passed: boolean;
  message: string;
}

export interface MarketAccuracyOut {
  market: string;
  brier: number | null;
  rps: number | null;
  log_loss: number | null;
  hit_rate: number | null;
  sample_size: number;
  badge: QualityBadgeOut;
  gates: GateStatusOut[];
  source: string;
}

export interface AccuracyOverviewOut {
  markets: MarketAccuracyOut[];
  total_settled: number;
  source: string;
}

export interface MarketAccuracyDetailOut {
  market: string;
  reliability: ReliabilityDiagramOut;
  calibration_bins: CalibrationBinOut[];
  brier: number | null;
  rps: number | null;
  log_loss: number | null;
  hit_rate: number | null;
  sample_size: number;
  badge: QualityBadgeOut;
  gates: GateStatusOut[];
  source: string;
}

export interface ModelVsMarketItemOut {
  market: string;
  model_metric: number;
  bookmaker_metric: number;
  metric_name: string;
  sample_size: number;
}

export interface ModelVsMarketOut {
  comparisons: ModelVsMarketItemOut[];
  source: string;
}
