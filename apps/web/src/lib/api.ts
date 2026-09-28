/** Typed fetch wrapper for the Football Predictor API. */

import type {
  FixtureOut,
  LeagueDetailOut,
  LeagueOut,
  MatchDetailOut,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://api:8000";

async function fetchApi<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { next: { revalidate: 900 } });
  if (!res.ok) {
    throw new Error(`API ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function getLeagues(): Promise<LeagueOut[]> {
  return fetchApi<LeagueOut[]>("/api/v1/leagues");
}

export async function getLeague(code: string): Promise<LeagueDetailOut> {
  return fetchApi<LeagueDetailOut>(`/api/v1/leagues/${code}`);
}

export async function getFixtures(
  date?: string,
  league?: string,
): Promise<FixtureOut[]> {
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  if (league) params.set("league", league);
  const qs = params.toString();
  return fetchApi<FixtureOut[]>(`/api/v1/fixtures${qs ? `?${qs}` : ""}`);
}

export async function getMatch(id: number): Promise<MatchDetailOut> {
  return fetchApi<MatchDetailOut>(`/api/v1/matches/${id}`);
}

export async function getNextFixtureDate(): Promise<string | null> {
  const data = await fetchApi<{ date: string | null }>("/api/v1/fixtures/next-date");
  return data.date;
}
