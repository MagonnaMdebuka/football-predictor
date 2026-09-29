/** 7x7 score probability heatmap with colour legend. */
"use client";

interface ScoreGridProps {
  grid: number[][];
}

const INTENSITY_LEVELS = [
  { min: 0.8, bg: "bg-emerald-600", label: "High" },
  { min: 0.6, bg: "bg-emerald-700", label: "" },
  { min: 0.4, bg: "bg-emerald-800", label: "Med" },
  { min: 0.2, bg: "bg-emerald-900", label: "" },
  { min: 0.05, bg: "bg-emerald-950", label: "Low" },
  { min: 0, bg: "bg-zinc-900", label: "—" },
];

function cellColour(prob: number, maxProb: number): string {
  if (maxProb === 0) return "bg-zinc-900";
  const intensity = prob / maxProb;
  for (const level of INTENSITY_LEVELS) {
    if (intensity > level.min) return level.bg;
  }
  return "bg-zinc-900";
}

export function ScoreGrid({ grid }: ScoreGridProps) {
  const maxProb = Math.max(...grid.flat());

  // Find top 3 cells
  const cells: { i: number; j: number; prob: number }[] = [];
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[i].length; j++) {
      cells.push({ i, j, prob: grid[i][j] });
    }
  }
  cells.sort((a, b) => b.prob - a.prob);
  const top3 = new Set(cells.slice(0, 3).map((c) => `${c.i}-${c.j}`));

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-3 sm:p-4 max-w-[500px] mx-auto">
      <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-3">
        Score Probability Grid
      </h3>

      {/* Grid table — scales proportionally on mobile */}
      <div className="overflow-x-auto">
        <table className="w-full text-xs" style={{ tableLayout: "fixed" }}>
          <thead>
            <tr>
              <th className="w-[12%] h-7 text-zinc-500 text-center"></th>
              {Array.from({ length: Math.min(grid[0]?.length ?? 0, 7) }, (_, j) => (
                <th key={j} className="h-7 text-center text-zinc-500 font-normal">
                  {j}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.slice(0, 7).map((row, i) => (
              <tr key={i}>
                <td className="text-center text-zinc-500 font-medium h-9 sm:h-11">{i}</td>
                {row.slice(0, 7).map((prob, j) => {
                  const isTop = top3.has(`${i}-${j}`);
                  const pct = (prob * 100).toFixed(1);
                  return (
                    <td
                      key={j}
                      className={`h-9 sm:h-11 text-center rounded-sm ${cellColour(prob, maxProb)} ${
                        isTop ? "ring-1 ring-zinc-400" : ""
                      }`}
                      title={`${i}-${j}: ${pct}%`}
                    >
                      {isTop && (
                        <div className="leading-tight">
                          <div className="font-bold text-white text-[10px]">{i}-{j}</div>
                          <div className="text-zinc-300 text-[9px]">{pct}%</div>
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Colour legend */}
      <div className="flex items-center justify-between mt-3 pt-3 border-t border-zinc-800">
        <span className="text-xs text-zinc-500">Probability</span>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-zinc-500">Low</span>
          <div className="w-4 h-3 rounded-sm bg-zinc-900 border border-zinc-800" />
          <div className="w-4 h-3 rounded-sm bg-emerald-950" />
          <div className="w-4 h-3 rounded-sm bg-emerald-900" />
          <div className="w-4 h-3 rounded-sm bg-emerald-800" />
          <div className="w-4 h-3 rounded-sm bg-emerald-700" />
          <div className="w-4 h-3 rounded-sm bg-emerald-600" />
          <span className="text-[10px] text-zinc-500">High</span>
        </div>
      </div>

      <p className="text-xs text-zinc-600 mt-2">
        Rows = home goals, columns = away goals. Top 3 highlighted.
      </p>
    </div>
  );
}
