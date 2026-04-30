import React from "react";

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value ?? "-"}</div>
    </div>
  );
}

export default function StatsPanel({ summary }) {
  const solution = summary?.solution || {};
  const agent = summary?.agent || {};

  return (
    <div className="stats-grid">
      <StatCard label="Run" value={summary?.run_id} />
      <StatCard label="Nodes" value={solution["number_of_nodes"]} />
      <StatCard label="Links" value={solution["number_of_links"]} />
      <StatCard label="Agents" value={solution["number_of_agents"]} />
      <StatCard label="CPU Time (s)" value={solution["CPU running time"]} />
      <StatCard label="Agent Travel Time" value={agent.travel_time ?? "fallback in backend"} />
    </div>
  );
}

