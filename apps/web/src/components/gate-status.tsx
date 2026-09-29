/** Pass/fail indicator row for publication gates. */

import type { GateStatusOut } from "@/lib/types";

const GATE_LABELS: Record<string, string> = {
  min_settled: "Sample size",
  max_decile_error: "Decile error",
  beats_baseline: "Beats baseline",
  direct_data: "Direct data",
};

interface GateStatusProps {
  gates: GateStatusOut[];
}

export function GateStatus({ gates }: GateStatusProps) {
  return (
    <div className="flex flex-wrap gap-3">
      {gates.map((gate) => (
        <div
          key={gate.name}
          className="flex items-center gap-1.5 text-xs"
          title={gate.message}
        >
          <span
            className={`w-4 h-4 flex items-center justify-center rounded text-xs font-bold ${
              gate.passed
                ? "bg-emerald-900 text-emerald-300"
                : "bg-red-900 text-red-300"
            }`}
          >
            {gate.passed ? "\u2713" : "\u2717"}
          </span>
          <span className="text-zinc-400">
            {GATE_LABELS[gate.name] ?? gate.name}
          </span>
        </div>
      ))}
    </div>
  );
}
