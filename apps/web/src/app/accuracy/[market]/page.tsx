import { notFound } from "next/navigation";
import Link from "next/link";
import { getMarketAccuracy } from "@/lib/api";
import { QualityBadge } from "@/components/quality-badge";
import { GateStatus } from "@/components/gate-status";
import { ReliabilityDiagram } from "@/components/reliability-diagram";

export const revalidate = 900;

const MARKET_LABELS: Record<string, string> = {
  match_result_home: "1X2 Home Win",
  match_result_draw: "1X2 Draw",
  match_result_away: "1X2 Away Win",
  "over_under_2.5_over": "Over 2.5 Goals",
  "over_under_2.5_under": "Under 2.5 Goals",
  "over_under_1.5_over": "Over 1.5 Goals",
  "over_under_1.5_under": "Under 1.5 Goals",
  "over_under_3.5_over": "Over 3.5 Goals",
  "over_under_3.5_under": "Under 3.5 Goals",
  btts_yes: "Both Teams to Score — Yes",
  btts_no: "Both Teams to Score — No",
};

const BADGE_EXPLANATIONS: Record<string, string> = {
  green:
    "This market has 3+ seasons of direct calibration data, providing reliable probability estimates.",
  amber:
    "This market has 1-2 seasons of data or uses cross-competition inference. Estimates are reasonable but may improve with more data.",
  grey:
    "Insufficient data for reliable calibration. Treat probability estimates with caution.",
};

interface PageProps {
  params: Promise<{ market: string }>;
}

export default async function MarketAccuracyPage({ params }: PageProps) {
  const { market } = await params;

  let data;
  try {
    data = await getMarketAccuracy(market);
  } catch {
    notFound();
  }

  const label = MARKET_LABELS[market] ?? market;

  return (
    <div>
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-zinc-500 mb-6">
        <Link href="/accuracy" className="hover:text-zinc-300">
          Accuracy
        </Link>
        <span>/</span>
        <span className="text-zinc-200">{label}</span>
      </div>

      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-xl sm:text-2xl font-bold text-zinc-100">{label}</h1>
        <QualityBadge badge={data.badge.badge} showLabel />
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mb-6 md:mb-8">
        {data.brier !== null && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <p className="text-xs text-zinc-500 uppercase tracking-wider">Brier Score</p>
            <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
              {data.brier.toFixed(4)}
            </p>
          </div>
        )}
        {data.rps !== null && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <p className="text-xs text-zinc-500 uppercase tracking-wider">RPS</p>
            <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
              {data.rps.toFixed(4)}
            </p>
          </div>
        )}
        {data.log_loss !== null && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <p className="text-xs text-zinc-500 uppercase tracking-wider">Log Loss</p>
            <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
              {data.log_loss.toFixed(4)}
            </p>
          </div>
        )}
        {data.hit_rate !== null && (
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
            <p className="text-xs text-zinc-500 uppercase tracking-wider">Hit Rate</p>
            <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
              {(data.hit_rate * 100).toFixed(1)}%
            </p>
          </div>
        )}
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <p className="text-xs text-zinc-500 uppercase tracking-wider">Sample</p>
          <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
            {data.sample_size.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Reliability diagram */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 mb-8">
        <h2 className="text-sm font-semibold text-zinc-200 mb-4">Reliability Diagram</h2>
        <ReliabilityDiagram bins={data.calibration_bins} />
        <div className="flex gap-6 mt-3 text-xs text-zinc-500">
          <span>
            Max decile error:{" "}
            <span className="text-zinc-300">
              {(data.reliability.calibration_error * 100).toFixed(1)}pp
            </span>
          </span>
          <span>
            Mean cal. error:{" "}
            <span className="text-zinc-300">
              {(data.reliability.mean_calibration_error * 100).toFixed(1)}pp
            </span>
          </span>
        </div>
      </div>

      {/* Calibration bins table */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden mb-8">
        <div className="px-4 py-3 border-b border-zinc-800">
          <h2 className="text-sm font-semibold text-zinc-200">Calibration Bins</h2>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 text-zinc-400 text-xs">
              <th className="text-left px-4 py-2 font-medium">Bin</th>
              <th className="text-right px-4 py-2 font-medium">Predicted</th>
              <th className="text-right px-4 py-2 font-medium">Observed</th>
              <th className="text-right px-4 py-2 font-medium">Error</th>
              <th className="text-right px-4 py-2 font-medium">Count</th>
            </tr>
          </thead>
          <tbody>
            {data.calibration_bins.map((bin, i) => {
              const error =
                bin.sample_size > 0
                  ? Math.abs(bin.predicted_frequency - bin.observed_frequency)
                  : 0;
              return (
                <tr key={i} className="border-b border-zinc-800/50">
                  <td className="px-4 py-2 text-zinc-300">
                    {(bin.bin_lower * 100).toFixed(0)}–{(bin.bin_upper * 100).toFixed(0)}%
                  </td>
                  <td className="text-right px-4 py-2 text-zinc-300 tabular-nums">
                    {(bin.predicted_frequency * 100).toFixed(1)}%
                  </td>
                  <td className="text-right px-4 py-2 text-zinc-300 tabular-nums">
                    {(bin.observed_frequency * 100).toFixed(1)}%
                  </td>
                  <td
                    className={`text-right px-4 py-2 tabular-nums ${
                      error > 0.05 ? "text-amber-400" : "text-zinc-500"
                    }`}
                  >
                    {(error * 100).toFixed(1)}pp
                  </td>
                  <td className="text-right px-4 py-2 text-zinc-500 tabular-nums">
                    {bin.sample_size}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid gap-6 lg:grid-cols-2 mb-8">
        {/* Gate status */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <h2 className="text-sm font-semibold text-zinc-200 mb-3">Publication Gates</h2>
          <GateStatus gates={data.gates} />
        </div>

        {/* Badge explanation */}
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
          <h2 className="text-sm font-semibold text-zinc-200 mb-3">Quality Assessment</h2>
          <div className="flex items-start gap-2">
            <QualityBadge badge={data.badge.badge} showLabel />
            <p className="text-xs text-zinc-400">
              {BADGE_EXPLANATIONS[data.badge.badge] ?? ""}
            </p>
          </div>
          <div className="mt-2 text-xs text-zinc-500">
            <span>Seasons: {data.badge.n_seasons}</span>
            <span className="ml-4">
              Direct data: {data.badge.has_direct_data ? "Yes" : "No"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
