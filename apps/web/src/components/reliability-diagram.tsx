"use client";

/** Reliability diagram — predicted vs observed probability scatter plot. */

import type { CalibrationBinOut } from "@/lib/types";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface ReliabilityDiagramProps {
  bins: CalibrationBinOut[];
}

export function ReliabilityDiagram({ bins }: ReliabilityDiagramProps) {
  const data = bins
    .filter((b) => b.sample_size > 0)
    .map((b) => ({
      predicted: Math.round(b.predicted_frequency * 100),
      observed: Math.round(b.observed_frequency * 100),
      count: b.sample_size,
      label: `${(b.bin_lower * 100).toFixed(0)}-${(b.bin_upper * 100).toFixed(0)}%`,
    }));

  const perfectLine = [
    { predicted: 0, observed: 0 },
    { predicted: 100, observed: 100 },
  ];

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            type="number"
            dataKey="predicted"
            domain={[0, 100]}
            name="Predicted"
            unit="%"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            label={{
              value: "Predicted probability (%)",
              position: "bottom",
              offset: 15,
              fill: "#71717a",
              fontSize: 12,
            }}
          />
          <YAxis
            type="number"
            dataKey="observed"
            domain={[0, 100]}
            name="Observed"
            unit="%"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            label={{
              value: "Observed frequency (%)",
              angle: -90,
              position: "insideLeft",
              offset: 0,
              fill: "#71717a",
              fontSize: 12,
            }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{
              backgroundColor: "#18181b",
              border: "1px solid #3f3f46",
              borderRadius: "0.5rem",
              fontSize: 12,
            }}
            formatter={(value: number, name: string) => [
              `${value}%`,
              name === "observed" ? "Observed" : "Predicted",
            ]}
            labelFormatter={(label: number) => `Predicted: ${label}%`}
          />
          {/* Perfect calibration line */}
          <ReferenceLine
            segment={perfectLine as any}
            stroke="#52525b"
            strokeDasharray="6 3"
            label={{
              value: "Perfect",
              position: "insideTopRight",
              fill: "#52525b",
              fontSize: 10,
            }}
          />
          <Scatter
            data={data}
            fill="#3b82f6"
            stroke="#60a5fa"
            strokeWidth={1}
            r={6}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
