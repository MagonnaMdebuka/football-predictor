/** Collapsible market accordion sections. */
"use client";

import { useState } from "react";
import type { MarketOut } from "@/lib/types";

const GROUP_LABELS: Record<string, string> = {
  goals: "Goals",
  result: "Result",
  handicaps: "Handicaps",
  score: "Score",
  defence: "Defence",
  ht_goals: "Half-Time Goals",
  ht_result: "Half-Time Result",
  ht_score: "Half-Time Score",
  halftime: "Half-Time",
};

function formatSelection(market: string, selection: string): string {
  // Correct score: digit_digit → "digit-digit"
  if (/^\d+_\d+$/.test(selection)) {
    return selection.replace("_", "-");
  }

  // Double chance / draw-no-bet labels
  const dcMap: Record<string, string> = {
    home_draw: "Home or Draw",
    draw_away: "Draw or Away",
    home_away: "Home or Away",
  };
  if (dcMap[selection]) return dcMap[selection];

  // Winning margin: "home_1" → "Home by 1", "home_+3" → "Home by 3+"
  const marginMatch = selection.match(/^(home|away|draw)_(\+?\d+)$/);
  if (marginMatch) {
    if (marginMatch[1] === "draw") return "Draw";
    const side = marginMatch[1].charAt(0).toUpperCase() + marginMatch[1].slice(1);
    const num = marginMatch[2].replace("+", "");
    const suffix = marginMatch[2].includes("+") ? "+" : "";
    return `${side} by ${num}${suffix}`;
  }

  // Generic fallback: replace underscores with spaces, title-case
  return selection
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

interface MarketAccordionProps {
  markets: Record<string, MarketOut[]>;
}

function MarketRow({ m }: { m: MarketOut }) {
  const pct = Math.round(m.probability * 100);
  if (pct === 0) return null;
  return (
    <div className="flex items-center gap-3 py-2 border-b border-zinc-800/50 last:border-0">
      {/* Label */}
      <div className="flex items-center gap-1.5 min-w-0 flex-shrink-0 w-[40%]">
        <span className="text-sm text-zinc-300 truncate">{formatSelection(m.market, m.selection)}</span>
        {m.line != null && (
          <span className="text-xs text-zinc-500 whitespace-nowrap">({m.line})</span>
        )}
      </div>
      {/* Inline probability bar */}
      <div className="flex-1 min-w-0">
        <div className="h-2 w-full bg-zinc-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-600 rounded-full transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
      {/* Percentage + odds */}
      <div className="flex items-center gap-3 flex-shrink-0">
        <span className="text-sm font-medium text-zinc-100 w-10 text-right tabular-nums">{pct}%</span>
        {m.fair_odds && (
          <span className="text-xs text-zinc-500 w-12 text-right tabular-nums hidden sm:block" title="Fair odds">
            {m.fair_odds.toFixed(2)}
          </span>
        )}
      </div>
    </div>
  );
}

export function MarketAccordion({ markets }: MarketAccordionProps) {
  const [open, setOpen] = useState<Record<string, boolean>>({ goals: true, result: true });

  const groups = Object.entries(markets);
  if (groups.length === 0) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 text-center">
        <p className="text-zinc-500 text-sm">No market predictions available.</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {groups.map(([group, items]) => (
        <div key={group} className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
          <button
            onClick={() => setOpen((prev) => ({ ...prev, [group]: !prev[group] }))}
            className="w-full flex items-center justify-between px-4 py-3 hover:bg-zinc-800/50 transition-colors"
          >
            <span className="text-sm font-medium text-zinc-200">
              {GROUP_LABELS[group] ?? group}
            </span>
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-500">
                {items.filter(m => Math.round(m.probability * 100) > 0).length}
              </span>
              <span className="text-zinc-500 text-xs">
                {open[group] ? "▲" : "▼"}
              </span>
            </div>
          </button>
          {open[group] && (
            <div className="px-4 pb-3 border-t border-zinc-800/50">
              {items.map((m, i) => (
                <MarketRow key={`${m.market}-${m.selection}-${m.line}-${i}`} m={m} />
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
