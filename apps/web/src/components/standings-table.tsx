/** League standings table. */

import type { StandingsRowOut } from "@/lib/types";

interface StandingsTableProps {
  standings: StandingsRowOut[];
}

export function StandingsTable({ standings }: StandingsTableProps) {
  if (standings.length === 0) {
    return (
      <div className="p-4 text-center">
        <p className="text-zinc-500 text-sm">No standings data available.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="sticky top-0 z-10 bg-zinc-900">
          <tr className="border-b border-zinc-700 text-zinc-400 text-xs">
            <th className="text-left py-2.5 pl-4 pr-2 w-8">#</th>
            <th className="text-left py-2.5 pr-4">Team</th>
            <th className="text-center py-2.5 px-1.5 w-8">P</th>
            <th className="text-center py-2.5 px-1.5 w-8">W</th>
            <th className="text-center py-2.5 px-1.5 w-8">D</th>
            <th className="text-center py-2.5 px-1.5 w-8">L</th>
            <th className="text-center py-2.5 px-1.5 w-8 hidden sm:table-cell">GF</th>
            <th className="text-center py-2.5 px-1.5 w-8 hidden sm:table-cell">GA</th>
            <th className="text-center py-2.5 px-1.5 w-10">GD</th>
            <th className="text-center py-2.5 pr-4 pl-1.5 w-10 font-bold">Pts</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((row, i) => (
            <tr
              key={row.team}
              className={`border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors ${
                i % 2 === 0 ? "bg-zinc-900" : "bg-zinc-950"
              }`}
            >
              <td className="py-2 pl-4 pr-2 text-zinc-500 tabular-nums">{i + 1}</td>
              <td className="py-2 pr-4 font-medium text-zinc-100 truncate max-w-[120px] sm:max-w-none">{row.team}</td>
              <td className="text-center py-2 px-1.5 text-zinc-400 tabular-nums">{row.played}</td>
              <td className="text-center py-2 px-1.5 text-zinc-400 tabular-nums">{row.won}</td>
              <td className="text-center py-2 px-1.5 text-zinc-400 tabular-nums">{row.drawn}</td>
              <td className="text-center py-2 px-1.5 text-zinc-400 tabular-nums">{row.lost}</td>
              <td className="text-center py-2 px-1.5 text-zinc-400 tabular-nums hidden sm:table-cell">{row.goals_for}</td>
              <td className="text-center py-2 px-1.5 text-zinc-400 tabular-nums hidden sm:table-cell">{row.goals_against}</td>
              <td className="text-center py-2 px-1.5 text-zinc-300 tabular-nums">
                {row.goal_difference > 0 ? `+${row.goal_difference}` : row.goal_difference}
              </td>
              <td className="text-center py-2 pr-4 pl-1.5 font-bold text-zinc-100 tabular-nums">{row.points}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
