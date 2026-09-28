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
};

interface MarketAccordionProps {
  markets: Record<string, MarketOut[]>;
}

function MarketRow({ m }: { m: MarketOut }) {
  const pct = Math.round(m.probability * 100);
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-zinc-800/50 last:border-0">
      <div className="flex items-center gap-2">
        <span className="text-sm text-zinc-300">{m.selection}</span>
        {m.line != null && (
          <span className="text-xs text-zinc-500">({m.line})</span>
        )}
      </div>
      <div className="flex items-center gap-4">
        <span className="text-sm font-medium text-zinc-100 w-12 text-right">{pct}%</span>
        {m.fair_odds && (
          <span className="text-xs text-zinc-500 w-14 text-right" title="No margin odds">
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
    return <p className="text-zinc-500 text-sm">No market predictions available.</p>;
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
            <span className="text-zinc-500 text-xs">
              {open[group] ? "▲" : "▼"}
            </span>
          </button>
          {open[group] && (
            <div className="px-4 pb-3">
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
