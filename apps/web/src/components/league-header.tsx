/** League page header with name and country. */

import type { LeagueOut } from "@/lib/types";

interface LeagueHeaderProps {
  league: LeagueOut;
}

export function LeagueHeader({ league }: LeagueHeaderProps) {
  return (
    <div className="mb-6">
      <h1 className="text-xl sm:text-2xl font-bold text-zinc-100">{league.name}</h1>
      <p className="text-xs sm:text-sm text-zinc-500">{league.country}</p>
    </div>
  );
}
