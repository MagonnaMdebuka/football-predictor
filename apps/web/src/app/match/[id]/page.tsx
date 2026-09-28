import { notFound } from "next/navigation";
import { getMatch } from "@/lib/api";
import { MatchHeader } from "@/components/match-header";
import { PredictionHeadline } from "@/components/prediction-headline";
import { MarketAccordion } from "@/components/market-accordion";
import { ScoreGrid } from "@/components/score-grid";
import { MatchFooter } from "@/components/match-footer";

export const revalidate = 60;

interface PageProps {
  params: Promise<{ id: string }>;
}

export default async function MatchPage({ params }: PageProps) {
  const { id } = await params;
  const matchId = parseInt(id, 10);
  if (isNaN(matchId)) notFound();

  let match;
  try {
    match = await getMatch(matchId);
  } catch {
    notFound();
  }

  return (
    <div>
      <MatchHeader
        homeTeam={match.home_team}
        awayTeam={match.away_team}
        kickoff={match.kickoff_utc}
        status={match.status}
        referee={match.referee}
      />

      {match.prediction && (
        <PredictionHeadline prediction={match.prediction} />
      )}

      {!match.prediction && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 mb-6 text-center">
          <p className="text-zinc-500">No prediction available for this match.</p>
          <p className="text-zinc-600 text-sm mt-1">Run the predict command to generate predictions.</p>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2 mb-6">
        <div>
          <h2 className="text-lg font-semibold text-zinc-200 mb-3">Markets</h2>
          <MarketAccordion markets={match.markets} />
        </div>

        <div>
          {match.grid && <ScoreGrid grid={match.grid} />}
        </div>
      </div>

      <MatchFooter />
    </div>
  );
}
