import { notFound } from "next/navigation";
import { getLeague } from "@/lib/api";
import { LeagueHeader } from "@/components/league-header";
import { StandingsTable } from "@/components/standings-table";
import { FixtureCard } from "@/components/fixture-card";

export const revalidate = 900;

interface PageProps {
  params: Promise<{ code: string }>;
}

export default async function LeaguePage({ params }: PageProps) {
  const { code } = await params;

  let data;
  try {
    data = await getLeague(code);
  } catch {
    notFound();
  }

  return (
    <div>
      <LeagueHeader league={data.league} />

      <div className="grid gap-6 lg:gap-8 lg:grid-cols-[3fr_2fr]">
        {/* Fixtures — left / top on mobile */}
        <div>
          <h2 className="text-lg sm:text-xl font-semibold text-zinc-200 mb-3">Upcoming Fixtures</h2>
          {data.fixtures.length === 0 ? (
            <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4 text-center">
              <p className="text-zinc-500 text-sm">No upcoming fixtures.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {data.fixtures.map((f) => (
                <FixtureCard key={f.id} fixture={f} />
              ))}
            </div>
          )}
        </div>

        {/* Standings — right / below on mobile */}
        <div>
          <h2 className="text-lg sm:text-xl font-semibold text-zinc-200 mb-3">Standings</h2>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 overflow-hidden">
            <StandingsTable standings={data.standings} />
          </div>
        </div>
      </div>
    </div>
  );
}
