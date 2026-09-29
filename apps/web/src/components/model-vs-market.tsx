/** Side-by-side model vs bookmaker comparison. */

import type { ModelVsMarketItemOut } from "@/lib/types";

const MARKET_LABELS: Record<string, string> = {
  match_result_home: "Home win",
  match_result_draw: "Draw",
  match_result_away: "Away win",
};

interface ModelVsMarketProps {
  comparisons: ModelVsMarketItemOut[];
}

export function ModelVsMarket({ comparisons }: ModelVsMarketProps) {
  if (comparisons.length === 0) {
    return (
      <p className="text-zinc-500 text-sm">No comparison data available.</p>
    );
  }

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
      <div className="px-4 py-3 border-b border-zinc-800">
        <h3 className="text-sm font-semibold text-zinc-200">
          Model vs Bookmaker
        </h3>
        <p className="text-xs text-zinc-500 mt-0.5">
          Model probabilities vs overround-stripped bookmaker odds
        </p>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-400 text-xs">
            <th className="text-left px-4 py-2 font-medium">Selection</th>
            <th className="text-right px-4 py-2 font-medium">Model avg</th>
            <th className="text-right px-4 py-2 font-medium">Bookmaker avg</th>
            <th className="text-right px-4 py-2 font-medium">Diff</th>
            <th className="text-right px-4 py-2 font-medium">n</th>
          </tr>
        </thead>
        <tbody>
          {comparisons.map((c) => {
            const diff = c.model_metric - c.bookmaker_metric;
            return (
              <tr key={c.market} className="border-b border-zinc-800/50">
                <td className="px-4 py-2 text-zinc-200">
                  {MARKET_LABELS[c.market] ?? c.market}
                </td>
                <td className="text-right px-4 py-2 text-zinc-300 tabular-nums">
                  {(c.model_metric * 100).toFixed(1)}%
                </td>
                <td className="text-right px-4 py-2 text-zinc-300 tabular-nums">
                  {(c.bookmaker_metric * 100).toFixed(1)}%
                </td>
                <td
                  className={`text-right px-4 py-2 tabular-nums ${
                    Math.abs(diff) < 0.01
                      ? "text-zinc-500"
                      : diff > 0
                        ? "text-emerald-400"
                        : "text-red-400"
                  }`}
                >
                  {diff > 0 ? "+" : ""}
                  {(diff * 100).toFixed(1)}pp
                </td>
                <td className="text-right px-4 py-2 text-zinc-500 tabular-nums">
                  {c.sample_size.toLocaleString()}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
