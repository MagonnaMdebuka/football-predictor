/** 11x11 score probability heatmap. */
"use client";

interface ScoreGridProps {
  grid: number[][];
}

function cellColour(prob: number, maxProb: number): string {
  if (maxProb === 0) return "bg-zinc-900";
  const intensity = prob / maxProb;
  if (intensity > 0.8) return "bg-emerald-600";
  if (intensity > 0.6) return "bg-emerald-700";
  if (intensity > 0.4) return "bg-emerald-800";
  if (intensity > 0.2) return "bg-emerald-900";
  if (intensity > 0.05) return "bg-emerald-950";
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
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wide mb-3">
        Score Probability Grid
      </h3>
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="w-8 h-8 text-zinc-500"></th>
              {Array.from({ length: Math.min(grid[0]?.length ?? 0, 7) }, (_, j) => (
                <th key={j} className="w-10 h-8 text-center text-zinc-500">
                  {j}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.slice(0, 7).map((row, i) => (
              <tr key={i}>
                <td className="text-center text-zinc-500 font-medium">{i}</td>
                {row.slice(0, 7).map((prob, j) => {
                  const isTop = top3.has(`${i}-${j}`);
                  const pct = (prob * 100).toFixed(1);
                  return (
                    <td
                      key={j}
                      className={`w-10 h-10 text-center rounded-sm ${cellColour(prob, maxProb)} ${
                        isTop ? "ring-1 ring-zinc-400" : ""
                      }`}
                      title={`${i}-${j}: ${pct}%`}
                    >
                      {isTop && (
                        <div>
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
      <p className="text-xs text-zinc-600 mt-2">
        Rows = home goals, columns = away goals. Top 3 scorelines highlighted.
      </p>
    </div>
  );
}
