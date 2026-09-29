/** Match detail header with team names and referee. */

import type { RefereeStatsOut } from "@/lib/types";

interface MatchHeaderProps {
  homeTeam: string;
  awayTeam: string;
  kickoff: string;
  status: string;
  referee: RefereeStatsOut | null;
  ftHomeGoals?: number | null;
  ftAwayGoals?: number | null;
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
  }) + " UTC";
}

export function MatchHeader({ homeTeam, awayTeam, kickoff, status, referee, ftHomeGoals, ftAwayGoals }: MatchHeaderProps) {
  const isFinished = status === "finished" && ftHomeGoals != null && ftAwayGoals != null;

  return (
    <div className="text-center mb-6 md:mb-8">
      {/* Teams row — stacks on very narrow screens */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-4 md:gap-6 mb-2">
        <span className="text-lg sm:text-xl md:text-2xl font-bold text-zinc-100 truncate max-w-[160px] sm:max-w-none">
          {homeTeam}
        </span>
        {isFinished ? (
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-2xl sm:text-3xl font-bold text-zinc-100 tabular-nums">
              {ftHomeGoals} - {ftAwayGoals}
            </span>
            <span className="text-xs bg-zinc-700 text-zinc-300 px-1.5 py-0.5 rounded">FT</span>
          </div>
        ) : (
          <span className="text-zinc-500 text-base sm:text-lg flex-shrink-0">vs</span>
        )}
        <span className="text-lg sm:text-xl md:text-2xl font-bold text-zinc-100 truncate max-w-[160px] sm:max-w-none">
          {awayTeam}
        </span>
      </div>

      <p className="text-xs sm:text-sm text-zinc-400">{formatKickoff(kickoff)}</p>
      <p className="text-xs text-zinc-600 uppercase tracking-wide mt-1">
        {isFinished ? "Full Time" : status === "scheduled" ? "Kick-off" : status}
      </p>

      {referee ? (
        <div className="mt-3 inline-flex items-center gap-2 bg-zinc-800 rounded-full px-3 py-1 max-w-full">
          <span className="text-xs text-zinc-400 truncate">Referee: {referee.name}</span>
          <span className="text-xs bg-zinc-700 text-zinc-300 px-2 py-0.5 rounded-full whitespace-nowrap">
            {referee.avg_cards_per_match} cards/match
          </span>
        </div>
      ) : (
        <div className="mt-3 inline-flex items-center bg-zinc-800 rounded-full px-3 py-1">
          <span className="text-xs text-zinc-500">Referee: TBD</span>
        </div>
      )}
    </div>
  );
}
