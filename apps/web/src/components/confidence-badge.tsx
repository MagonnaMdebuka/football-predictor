/** Colour-coded confidence badge. */

const CONFIDENCE_STYLES: Record<string, string> = {
  very_high: "bg-emerald-900 text-emerald-300 border-emerald-700",
  high: "bg-blue-900 text-blue-300 border-blue-700",
  medium: "bg-amber-900 text-amber-300 border-amber-700",
  low: "bg-zinc-800 text-zinc-400 border-zinc-600",
};

const CONFIDENCE_LABELS: Record<string, string> = {
  very_high: "Very High",
  high: "High",
  medium: "Medium",
  low: "Low",
};

interface ConfidenceBadgeProps {
  confidence: string | null;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  if (!confidence) return null;
  const style = CONFIDENCE_STYLES[confidence] ?? CONFIDENCE_STYLES.low;
  const label = CONFIDENCE_LABELS[confidence] ?? confidence;

  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded border ${style}`}>
      {label}
    </span>
  );
}
