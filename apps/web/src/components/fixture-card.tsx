/** Fixture card showing team names, kickoff, prediction summary. */

import Link from "next/link";
import type { FixtureOut } from "@/lib/types";
import { ProbabilityBar } from "./probability-bar";
import { ConfidenceBadge } from "./confidence-badge";

interface FixtureCardProps {
  fixture: FixtureOut;
}

function formatKickoff(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
  }) + ", " + d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false });
}

export function FixtureCard({ fixture }: FixtureCardProps) {
  const pred = fixture.prediction;
  const isFinished = fixture.status === "finished" && fixture.ft_home_goals != null && fixture.ft_away_goals != null;

  return (
    <Link href={`/match/${fixture.id}`} className="block">
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 hover:border-zinc-600 hover:bg-zinc-800/50 transition-colors h-full">
        {/* League + kickoff header */}
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-zinc-500 uppercase tracking-wide truncate mr-2">
            {fixture.league_name ?? fixture.league_code}
          </span>
          <span className="text-xs text-zinc-500 whitespace-nowrap">{formatKickoff(fixture.kickoff_utc)}</span>
        </div>

        {/* Teams + score */}
        <div className="flex items-center justify-between gap-2 mb-3">
          <span className="font-medium text-sm md:text-base text-zinc-100 truncate flex-1 min-w-0">{fixture.home_team}</span>
          {isFinished ? (
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-lg font-bold text-zinc-100 tabular-nums">
                {fixture.ft_home_goals} - {fixture.ft_away_goals}
              </span>
              <span className="text-xs bg-zinc-700 text-zinc-300 px-1.5 py-0.5 rounded">FT</span>
            </div>
          ) : (
            <span className="text-zinc-500 text-sm flex-shrink-0">vs</span>
          )}
          <span className="font-medium text-sm md:text-base text-zinc-100 truncate flex-1 min-w-0 text-right">{fixture.away_team}</span>
        </div>

        {/* Prediction bar (scheduled only) */}
        {!isFinished && pred && (
          <>
            <ProbabilityBar home={pred.home_win_prob} draw={pred.draw_prob} away={pred.away_win_prob} />
            <div className="flex items-center justify-between mt-2 gap-2">
              <div className="flex items-center gap-2 min-w-0">
                {pred.most_likely_score && (
                  <span className="text-sm text-zinc-300 whitespace-nowrap">
                    Predicted: {pred.most_likely_score}
                  </span>
                )}
                <ConfidenceBadge confidence={pred.confidence} />
              </div>
              {pred.home_expected_goals != null && pred.away_expected_goals != null && (
                <span className="text-xs text-zinc-500 whitespace-nowrap">
                  xG {pred.home_expected_goals.toFixed(1)} - {pred.away_expected_goals.toFixed(1)}
                </span>
              )}
            </div>
          </>
        )}

        {!isFinished && !pred && (
          <p className="text-xs text-zinc-600 mt-1">No prediction available</p>
        )}
      </div>
    </Link>
  );
}
