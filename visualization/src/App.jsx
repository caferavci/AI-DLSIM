import React, { useEffect, useState } from "react";
import NetworkMap from "./components/NetworkMap";
import { getRunNetwork, getRunRoute, getRunSummary } from "./lib/api";

const RUN_ID = "query_pipeline";
const REFRESH_MS = 10000;

export default function App() {
  const [summary, setSummary] = useState(null);
  const [network, setNetwork] = useState([]);
  const [routePoints, setRoutePoints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [refreshTick, setRefreshTick] = useState(0);

  useEffect(() => {
    async function loadRunData() {
      setLoading(true);
      try {
        const [summaryData, networkData, routeData] = await Promise.all([
          getRunSummary(RUN_ID),
          getRunNetwork(RUN_ID, 1500),
          getRunRoute(RUN_ID),
        ]);
        setSummary(summaryData);
        setNetwork(networkData.links || []);
        setRoutePoints(routeData.route_points || []);
        setRefreshTick((n) => n + 1);
        setError("");
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }

    loadRunData();
    const timer = setInterval(loadRunData, REFRESH_MS);
    return () => clearInterval(timer);
  }, []);

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>AI-DLSIM Visualization</h1>
        <div className="toolbar">
          <span>Run: <strong>{RUN_ID}</strong></span>
          <span>Auto-refresh: {REFRESH_MS / 1000}s</span>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}
      {loading ? <div className="loading">Loading run data...</div> : null}

      <div className="panel answer-panel">
        <h3>Latest Query Result</h3>
        {summary?.query ? <p><strong>Prompt:</strong> {summary.query}</p> : null}
        <p className="answer-text">
          {summary?.final_answer || "Run the query pipeline once to populate a human-readable final answer."}
        </p>
      </div>

      <div className="grid-layout single-column">
        <NetworkMap
          links={network}
          routePoints={routePoints}
          mapKey={`map-${refreshTick}-${summary?.updated_at || "none"}`}
        />
      </div>
    </div>
  );
}

