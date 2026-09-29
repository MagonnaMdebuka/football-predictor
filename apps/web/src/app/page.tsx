import { Suspense } from "react";
import { getFixtures, getLeagues, getNextFixtureDate } from "@/lib/api";
import { FixtureCard } from "@/components/fixture-card";
import { DateStrip } from "@/components/date-strip";
import { LeagueChips } from "@/components/league-chips";

export const revalidate = 900;

interface PageProps {
  searchParams: Promise<{ date?: string; league?: string }>;
}

export default async function Home({ searchParams }: PageProps) {
  const params = await searchParams;
  const explicitDate = params.date;
  const league = params.league;
  const today = new Date().toISOString().slice(0, 10);

  let date = explicitDate ?? today;
  let fixtures;
  let leagues;
  let showingNextDate = false;

  try {
    [fixtures, leagues] = await Promise.all([
      getFixtures(date, league),
      getLeagues(),
    ]);

    // When no explicit date was set and today has no fixtures,
    // fall forward to the next future date that does.
    if (!explicitDate && fixtures.length === 0) {
      const nextDate = await getNextFixtureDate();
      if (nextDate && nextDate !== date && nextDate >= today) {
        fixtures = await getFixtures(nextDate, league);
        date = nextDate;
        showingNextDate = true;
      }
    }
  } catch {
    fixtures = [];
    leagues = [];
  }

  function formatDateHeading(iso: string): string {
    const d = new Date(iso + "T12:00:00");
    return d.toLocaleDateString("en-GB", {
      weekday: "short",
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  }

  const heading = showingNextDate
    ? `Next Fixtures — ${formatDateHeading(date)}`
    : explicitDate
      ? `Fixtures — ${formatDateHeading(date)}`
      : "Today\u2019s Fixtures";

  return (
    <div>
      <h1 className="text-2xl font-bold mb-1">{heading}</h1>
      {showingNextDate && (
        <p className="text-zinc-500 text-sm mb-4">
          No fixtures today — showing {date}
        </p>
      )}

      <Suspense fallback={<div className="h-16" />}>
        <div className="mb-4">
          <DateStrip selectedDate={date} />
        </div>
      </Suspense>

      <Suspense fallback={null}>
        <div className="mb-6">
          <LeagueChips leagues={leagues} />
        </div>
      </Suspense>

      {fixtures.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-zinc-500 text-lg">
            No fixtures scheduled for today.
          </p>
          <p className="text-zinc-600 text-sm mt-2">
            Browse a different date or check the league page.
          </p>
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {fixtures.map((f) => (
            <FixtureCard key={f.id} fixture={f} />
          ))}
        </div>
      )}
    </div>
  );
}
