/** Colour-coded quality badge for market accuracy. */

const BADGE_STYLES: Record<string, string> = {
  green: "bg-emerald-900 text-emerald-300 border-emerald-700",
  amber: "bg-amber-900 text-amber-300 border-amber-700",
  grey: "bg-zinc-800 text-zinc-400 border-zinc-600",
};

const BADGE_LABELS: Record<string, string> = {
  green: "Reliable",
  amber: "Limited",
  grey: "Insufficient",
};

interface QualityBadgeProps {
  badge: string;
  showLabel?: boolean;
}

export function QualityBadge({ badge, showLabel = false }: QualityBadgeProps) {
  const style = BADGE_STYLES[badge] ?? BADGE_STYLES.grey;
  const label = BADGE_LABELS[badge] ?? badge;

  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded border ${style}`}>
      <span
        className={`w-2 h-2 rounded-full ${
          badge === "green"
            ? "bg-emerald-400"
            : badge === "amber"
              ? "bg-amber-400"
              : "bg-zinc-500"
        }`}
      />
      {showLabel && label}
    </span>
  );
}
