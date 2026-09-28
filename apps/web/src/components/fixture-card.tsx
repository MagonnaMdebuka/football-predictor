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
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", hour12: false });
}

export function FixtureCard({ fixture }: FixtureCardProps) {
  const pred = fixture.prediction;

  return (
    <Link href={`/match/${fixture.id}`} className="block">
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 hover:border-zinc-600 transition-colors">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-zinc-500 uppercase tracking-wide">
            {fixture.league_code}
          </span>
          <span className="text-xs text-zinc-500">{formatKickoff(fixture.kickoff_utc)}</span>
        </div>

        <div className="flex items-center justify-between mb-3">
          <span className="font-medium text-zinc-100">{fixture.home_team}</span>
          <span className="text-zinc-500 text-sm">vs</span>
          <span className="font-medium text-zinc-100">{fixture.away_team}</span>
        </div>

        {pred && (
          <>
            <ProbabilityBar home={pred.home_win_prob} draw={pred.draw_prob} away={pred.away_win_prob} />
            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center gap-2">
                {pred.most_likely_score && (
                  <span className="text-sm text-zinc-300">
                    Predicted: {pred.most_likely_score}
                  </span>
                )}
                <ConfidenceBadge confidence={pred.confidence} />
              </div>
              {pred.home_expected_goals != null && pred.away_expected_goals != null && (
                <span className="text-xs text-zinc-500">
                  xG {pred.home_expected_goals.toFixed(1)} - {pred.away_expected_goals.toFixed(1)}
                </span>
              )}
            </div>
          </>
        )}

        {!pred && (
          <p className="text-xs text-zinc-600 mt-1">No prediction available</p>
        )}
      </div>
    </Link>
  );
}
