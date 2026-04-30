import React, { useMemo } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

export default function SpeedChart({ links }) {
  const chartData = useMemo(() => {
    if (!links?.length) {
      return null;
    }

    const sampled = links.slice(0, 20);
    return {
      labels: sampled.map((d) => d.link_id),
      datasets: [
        {
          label: "Speed",
          data: sampled.map((d) => d.speed ?? 0),
          backgroundColor: "rgba(60, 130, 255, 0.7)",
        },
      ],
    };
  }, [links]);

  if (!chartData) {
    return <div className="panel">No chart data yet.</div>;
  }

  return (
    <div className="panel">
      <h3>Link Speed Snapshot (first 20 links)</h3>
      <Bar
        data={chartData}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
          },
        }}
      />
    </div>
  );
}

