import Link from "next/link";
import { getAccuracyOverview, getModelVsMarket } from "@/lib/api";
import { QualityBadge } from "@/components/quality-badge";
import { GateStatus } from "@/components/gate-status";
import { ModelVsMarket } from "@/components/model-vs-market";

export const revalidate = 900;

const MARKET_LABELS: Record<string, string> = {
  match_result_home: "1X2 Home",
  match_result_draw: "1X2 Draw",
  match_result_away: "1X2 Away",
  match_result: "Match Result (RPS)",
  "over_under_2.5_over": "Over 2.5 Goals",
  "over_under_2.5_under": "Under 2.5 Goals",
  "over_under_1.5_over": "Over 1.5 Goals",
  "over_under_1.5_under": "Under 1.5 Goals",
  "over_under_3.5_over": "Over 3.5 Goals",
  "over_under_3.5_under": "Under 3.5 Goals",
  btts_yes: "BTTS Yes",
  btts_no: "BTTS No",
};

function sourceLabel(source: string, totalSettled: number): string {
  if (source === "backtest") {
    return `Based on ${totalSettled.toLocaleString()}-match backtest`;
  }
  return `Based on ${totalSettled.toLocaleString()} live predictions`;
}

export default async function AccuracyPage() {
  let overview;
  let modelVsMarket;
  try {
    [overview, modelVsMarket] = await Promise.all([
      getAccuracyOverview(),
      getModelVsMarket(),
    ]);
  } catch {
    return (
      <div className="text-center py-12">
        <h1 className="text-2xl font-bold text-zinc-200 mb-2">Model Accuracy</h1>
        <p className="text-zinc-500">
          Accuracy data is not yet available. Run the calibration bootstrap first.
        </p>
        <p className="text-zinc-600 text-sm mt-2">
          <code>./run.sh calibrate --league E0</code>
        </p>
      </div>
    );
  }

  const hasData = overview.markets.length > 0;

  // Summary stats across all markets
  const brierMarkets = overview.markets.filter((m) => m.brier !== null);
  const avgBrier =
    brierMarkets.length > 0
      ? brierMarkets.reduce((sum, m) => sum + (m.brier ?? 0), 0) / brierMarkets.length
      : null;
  const rpsMarkets = overview.markets.filter((m) => m.rps !== null);
  const avgRps =
    rpsMarkets.length > 0
      ? rpsMarkets.reduce((sum, m) => sum + (m.rps ?? 0), 0) / rpsMarkets.length
      : null;
  const avgHitRate =
    overview.markets.length > 0
      ? overview.markets
          .filter((m) => m.hit_rate !== null)
          .reduce((sum, m) => sum + (m.hit_rate ?? 0), 0) /
        overview.markets.filter((m) => m.hit_rate !== null).length
      : null;

  return (
    <div>
      {/* Header */}
      <div className="mb-6 md:mb-8">
        <h1 className="text-xl sm:text-2xl font-bold text-zinc-100">Model Accuracy</h1>
        <p className="text-sm text-zinc-500 mt-1">
          {hasData
            ? sourceLabel(overview.source, overview.total_settled)
            : "No data available"}
        </p>
      </div>

      {!hasData && (
        <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-6 text-center">
          <p className="text-zinc-500">
            No accuracy data yet. Run calibration bootstrap to populate.
          </p>
        </div>
      )}

      {hasData && (
        <>
          {/* Summary cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mb-6 md:mb-8">
            {avgBrier !== null && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
                <p className="text-xs text-zinc-500 uppercase tracking-wider">Avg Brier</p>
                <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
                  {avgBrier.toFixed(4)}
                </p>
              </div>
            )}
            {avgRps !== null && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
                <p className="text-xs text-zinc-500 uppercase tracking-wider">Avg RPS</p>
                <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
                  {avgRps.toFixed(4)}
                </p>
              </div>
            )}
            {avgHitRate !== null && (
              <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
                <p className="text-xs text-zinc-500 uppercase tracking-wider">Hit Rate</p>
                <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
                  {(avgHitRate * 100).toFixed(1)}%
                </p>
              </div>
            )}
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
              <p className="text-xs text-zinc-500 uppercase tracking-wider">Sample</p>
              <p className="text-2xl font-bold text-zinc-100 tabular-nums mt-1">
                {overview.total_settled.toLocaleString()}
              </p>
            </div>
          </div>

          {/* Market table */}
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden mb-8">
            <div className="px-4 py-3 border-b border-zinc-800">
              <h2 className="text-sm font-semibold text-zinc-200">Per-Market Accuracy</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-400 text-xs">
                    <th className="text-left px-4 py-2 font-medium">Market</th>
                    <th className="text-center px-4 py-2 font-medium">Quality</th>
                    <th className="text-right px-4 py-2 font-medium">Brier / RPS</th>
                    <th className="text-right px-4 py-2 font-medium">Hit Rate</th>
                    <th className="text-right px-4 py-2 font-medium">n</th>
                    <th className="text-center px-4 py-2 font-medium">Gates</th>
                  </tr>
                </thead>
                <tbody>
                  {overview.markets.map((m) => (
                    <tr key={m.market} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
                      <td className="px-4 py-2">
                        <Link
                          href={`/accuracy/${m.market}`}
                          className="text-blue-400 hover:text-blue-300"
                        >
                          {MARKET_LABELS[m.market] ?? m.market}
                        </Link>
                      </td>
                      <td className="px-4 py-2 text-center">
                        <QualityBadge badge={m.badge.badge} showLabel />
                      </td>
                      <td className="text-right px-4 py-2 text-zinc-300 tabular-nums">
                        {m.brier !== null
                          ? m.brier.toFixed(4)
                          : m.rps !== null
                            ? m.rps.toFixed(4)
                            : "—"}
                      </td>
                      <td className="text-right px-4 py-2 text-zinc-300 tabular-nums">
                        {m.hit_rate !== null ? `${(m.hit_rate * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="text-right px-4 py-2 text-zinc-500 tabular-nums">
                        {m.sample_size.toLocaleString()}
                      </td>
                      <td className="px-4 py-2">
                        <GateStatus gates={m.gates} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model vs Market */}
          <ModelVsMarket comparisons={modelVsMarket.comparisons} />
        </>
      )}
    </div>
  );
}
