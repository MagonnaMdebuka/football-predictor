/** Match detail header with team names and referee. */

import type { RefereeStatsOut } from "@/lib/types";

interface MatchHeaderProps {
  homeTeam: string;
  awayTeam: string;
  kickoff: string;
  status: string;
  referee: RefereeStatsOut | null;
}

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function MatchHeader({ homeTeam, awayTeam, kickoff, status, referee }: MatchHeaderProps) {
  return (
    <div className="text-center mb-8">
      <div className="flex items-center justify-center gap-6 mb-2">
        <span className="text-2xl font-bold text-zinc-100">{homeTeam}</span>
        <span className="text-zinc-500 text-lg">vs</span>
        <span className="text-2xl font-bold text-zinc-100">{awayTeam}</span>
      </div>
      <p className="text-sm text-zinc-400">{formatKickoff(kickoff)}</p>
      <p className="text-xs text-zinc-600 uppercase tracking-wide mt-1">{status}</p>

      {referee && (
        <div className="mt-3 inline-flex items-center gap-2 bg-zinc-800 rounded-full px-3 py-1">
          <span className="text-xs text-zinc-400">Referee: {referee.name}</span>
          <span className="text-xs bg-zinc-700 text-zinc-300 px-2 py-0.5 rounded-full">
            Historical avg: {referee.avg_cards_per_match} cards/match
          </span>
        </div>
      )}
    </div>
  );
}
