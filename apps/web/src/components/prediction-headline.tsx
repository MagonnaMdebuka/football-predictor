/** Prediction headline with 1X2 bar, predicted score, xG, confidence. */

import type { PredictionSummary } from "@/lib/types";
import { ProbabilityBar } from "./probability-bar";
import { ConfidenceBadge } from "./confidence-badge";

interface PredictionHeadlineProps {
  prediction: PredictionSummary;
}

export function PredictionHeadline({ prediction }: PredictionHeadlineProps) {
  const p = prediction;

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 md:p-6 mb-6">
      <h2 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-4">
        Prediction
      </h2>

      <ProbabilityBar home={p.home_win_prob} draw={p.draw_prob} away={p.away_win_prob} size="lg" />

      {p.scoreline_note && (
        <div className="mt-4 p-3 rounded-md bg-blue-950/50 border border-blue-800/50 text-xs sm:text-sm text-blue-200">
          {p.scoreline_note}
        </div>
      )}

      <div className="grid grid-cols-3 gap-2 sm:gap-4 mt-4 sm:mt-6 text-center">
        <div>
          <p className="text-xl sm:text-2xl font-bold text-zinc-100 tabular-nums">{p.most_likely_score ?? "—"}</p>
          <p className="text-xs text-zinc-500 mt-1">Predicted Score</p>
        </div>
        <div>
          {p.home_expected_goals != null && p.away_expected_goals != null ? (
            <p className="text-xl sm:text-2xl font-bold text-zinc-100 tabular-nums">
              {p.home_expected_goals.toFixed(2)} – {p.away_expected_goals.toFixed(2)}
            </p>
          ) : (
            <p className="text-xl sm:text-2xl font-bold text-zinc-500">—</p>
          )}
          <p className="text-xs text-zinc-500 mt-1">Expected Goals</p>
        </div>
        <div>
          <div className="flex justify-center">
            <ConfidenceBadge confidence={p.confidence} />
          </div>
          <p className="text-xs text-zinc-500 mt-1">Confidence</p>
        </div>
      </div>
    </div>
  );
}
