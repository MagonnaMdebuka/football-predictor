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

      <div className="grid gap-8 lg:grid-cols-2">
        <div>
          <h2 className="text-lg font-semibold text-zinc-200 mb-3">Upcoming Fixtures</h2>
          {data.fixtures.length === 0 ? (
            <p className="text-zinc-500 text-sm">No upcoming fixtures.</p>
          ) : (
            <div className="space-y-3">
              {data.fixtures.map((f) => (
                <FixtureCard key={f.id} fixture={f} />
              ))}
            </div>
          )}
        </div>

        <div>
          <h2 className="text-lg font-semibold text-zinc-200 mb-3">Standings</h2>
          <StandingsTable standings={data.standings} />
        </div>
      </div>
    </div>
  );
}
