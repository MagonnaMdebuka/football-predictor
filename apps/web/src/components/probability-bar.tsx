/** Stacked horizontal probability bar for 1X2 outcomes. */

interface ProbabilityBarProps {
  home: number;
  draw: number;
  away: number;
  size?: "sm" | "lg";
}

export function ProbabilityBar({ home, draw, away, size = "sm" }: ProbabilityBarProps) {
  const h = Math.round(home * 100);
  const d = Math.round(draw * 100);
  const a = 100 - h - d;
  const height = size === "lg" ? "h-8" : "h-5";

  return (
    <div className="w-full min-w-0">
      <div className={`flex ${height} rounded-md overflow-hidden text-xs font-medium`}>
        <div
          className="bg-emerald-600 flex items-center justify-center text-white"
          style={{ width: `${h}%` }}
        >
          {h > 8 && `${h}%`}
        </div>
        <div
          className="bg-zinc-500 flex items-center justify-center text-white"
          style={{ width: `${d}%` }}
        >
          {d > 8 && `${d}%`}
        </div>
        <div
          className="bg-blue-600 flex items-center justify-center text-white"
          style={{ width: `${a}%` }}
        >
          {a > 8 && `${a}%`}
        </div>
      </div>
      {size === "lg" && (
        <div className="flex justify-between text-xs text-zinc-400 mt-1">
          <span>Home {h}%</span>
          <span>Draw {d}%</span>
          <span>Away {a}%</span>
        </div>
      )}
    </div>
  );
}
