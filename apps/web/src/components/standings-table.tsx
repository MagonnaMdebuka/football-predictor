/** League standings table. */

import type { StandingsRowOut } from "@/lib/types";

interface StandingsTableProps {
  standings: StandingsRowOut[];
}

export function StandingsTable({ standings }: StandingsTableProps) {
  if (standings.length === 0) {
    return <p className="text-zinc-500 text-sm">No standings data available.</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-400 text-xs">
            <th className="text-left py-2 pr-4">#</th>
            <th className="text-left py-2 pr-4">Team</th>
            <th className="text-center py-2 px-2">P</th>
            <th className="text-center py-2 px-2">W</th>
            <th className="text-center py-2 px-2">D</th>
            <th className="text-center py-2 px-2">L</th>
            <th className="text-center py-2 px-2">GF</th>
            <th className="text-center py-2 px-2">GA</th>
            <th className="text-center py-2 px-2">GD</th>
            <th className="text-center py-2 px-2 font-bold">Pts</th>
          </tr>
        </thead>
        <tbody>
          {standings.map((row, i) => (
            <tr key={row.team} className="border-b border-zinc-800/50 hover:bg-zinc-800/30">
              <td className="py-2 pr-4 text-zinc-500">{i + 1}</td>
              <td className="py-2 pr-4 font-medium text-zinc-100">{row.team}</td>
              <td className="text-center py-2 px-2 text-zinc-400">{row.played}</td>
              <td className="text-center py-2 px-2 text-zinc-400">{row.won}</td>
              <td className="text-center py-2 px-2 text-zinc-400">{row.drawn}</td>
              <td className="text-center py-2 px-2 text-zinc-400">{row.lost}</td>
              <td className="text-center py-2 px-2 text-zinc-400">{row.goals_for}</td>
              <td className="text-center py-2 px-2 text-zinc-400">{row.goals_against}</td>
              <td className="text-center py-2 px-2 text-zinc-300">
                {row.goal_difference > 0 ? `+${row.goal_difference}` : row.goal_difference}
              </td>
              <td className="text-center py-2 px-2 font-bold text-zinc-100">{row.points}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
