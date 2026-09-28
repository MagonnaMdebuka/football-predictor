/** Filter chips for selecting a league. */
"use client";

import { useRouter, useSearchParams } from "next/navigation";
import type { LeagueOut } from "@/lib/types";

interface LeagueChipsProps {
  leagues: LeagueOut[];
}

export function LeagueChips({ leagues }: LeagueChipsProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selected = searchParams.get("league") ?? "";

  function handleClick(code: string) {
    const params = new URLSearchParams(searchParams.toString());
    if (code === selected) {
      params.delete("league");
    } else {
      params.set("league", code);
    }
    router.push(`/?${params.toString()}`);
  }

  return (
    <div className="flex gap-2 flex-wrap">
      {leagues.map((lg) => (
        <button
          key={lg.code}
          onClick={() => handleClick(lg.code)}
          className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
            lg.code === selected
              ? "bg-zinc-100 text-zinc-900"
              : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
          }`}
        >
          {lg.name}
        </button>
      ))}
    </div>
  );
}
