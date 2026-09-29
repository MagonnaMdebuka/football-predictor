/** Scrollable date strip for selecting a fixture date. */
"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

function getDates(centre: string, range: number): string[] {
  const dates: string[] = [];
  const centreDate = new Date(centre + "T12:00:00");
  for (let i = -3; i <= range; i++) {
    const d = new Date(centreDate);
    d.setDate(d.getDate() + i);
    dates.push(d.toISOString().slice(0, 10));
  }
  return dates;
}

interface DateStripProps {
  selectedDate?: string;
}

export function DateStrip({ selectedDate }: DateStripProps) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selected = selectedDate ?? searchParams.get("date") ?? new Date().toISOString().slice(0, 10);
  const dates = getDates(selected, 7);
  const containerRef = useRef<HTMLDivElement>(null);
  const selectedRef = useRef<HTMLButtonElement>(null);

  // Use selectedDate as initial "today" for stable SSR, then set the real
  // value client-side to avoid hydration mismatch.
  const [today, setToday] = useState(selectedDate ?? "");
  useEffect(() => {
    setToday(new Date().toISOString().slice(0, 10));
  }, []);

  // Scroll the selected/today button into view on mount
  useEffect(() => {
    if (selectedRef.current) {
      selectedRef.current.scrollIntoView({
        inline: "center",
        block: "nearest",
        behavior: "smooth",
      });
    }
  }, [selected]);

  function formatDay(iso: string): { day: string; label: string } {
    const d = new Date(iso + "T12:00:00");
    const label = iso === today ? "Today" : d.toLocaleDateString("en-GB", { weekday: "short" });
    const day = d.getDate().toString();
    return { day, label };
  }

  return (
    <div
      ref={containerRef}
      className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide md:justify-center"
    >
      {dates.map((iso) => {
        const { day, label } = formatDay(iso);
        const active = iso === selected;
        return (
          <button
            key={iso}
            ref={active ? selectedRef : undefined}
            onClick={() => {
              const params = new URLSearchParams(searchParams.toString());
              params.set("date", iso);
              router.push(`/?${params.toString()}`);
            }}
            className={`flex-shrink-0 flex flex-col items-center w-14 py-2 rounded-lg text-xs transition-colors ${
              active
                ? "bg-zinc-700 text-zinc-100 border border-zinc-500"
                : "bg-zinc-900 text-zinc-400 border border-zinc-800 hover:border-zinc-600"
            }`}
          >
            <span className="font-medium">{label}</span>
            <span className="text-lg font-bold">{day}</span>
          </button>
        );
      })}
    </div>
  );
}
