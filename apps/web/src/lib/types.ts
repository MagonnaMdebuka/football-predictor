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
