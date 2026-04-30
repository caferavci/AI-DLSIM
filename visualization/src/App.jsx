import React, { useEffect, useState } from "react";
import NetworkMap from "./components/NetworkMap";
import { getEngineView, getRunEngines, getRunSummary, runQuery } from "./lib/api";

const RUN_ID = "query_pipeline";
const REFRESH_MS = 10000;

export default function App() {
  const [summary, setSummary] = useState(null);
  const [engineMeta, setEngineMeta] = useState(null);
  const [leftView, setLeftView] = useState(null);
  const [rightView, setRightView] = useState(null);
  const [loading, setLoading] = useState(false);
  const [runningQuery, setRunningQuery] = useState(false);
  const [runError, setRunError] = useState("");
  const [submittedQuery, setSubmittedQuery] = useState("");
  const [sessionStarted, setSessionStarted] = useState(false);
  const [error, setError] = useState("");
  const [refreshTick, setRefreshTick] = useState(0); // force remount maps
  const [queryText, setQueryText] = useState(
    "What is the travel time from Cornell University to Ithaca Commons at 8:30 AM?"
  );

  const leftRouteTT = leftView?.route?.travel_time;
  const rightRouteTT = rightView?.route?.travel_time;
  const ttDelta =
    leftRouteTT != null && rightRouteTT != null ? Number((rightRouteTT - leftRouteTT).toFixed(2)) : null;
  const routeAvailable = (leftView?.route?.route_points || []).length > 1;
  const dlsimStatus = routeAvailable ? "Route available" : "No feasible agent output";
  const dtaliteIncompatible =
    String(summary?.final_answer || "").includes("Bad CPU type in executable") ||
    (engineMeta?.engines?.dtalite?.exists &&
      !engineMeta?.engines?.dtalite?.has_agent &&
      !engineMeta?.engines?.dtalite?.has_link_performance);
  const dtaliteStatus = dtaliteIncompatible ? "Binary incompatible" : "Available";
  const overallStatus = routeAvailable
    ? dtaliteIncompatible
      ? "Partial"
      : "Success"
    : "Failed";
  const displayStatus = runningQuery ? "In Progress" : runError ? "Failed" : overallStatus;

  async function loadRunData() {
    setLoading(true);
    try {
      const [summaryData, enginesData, dlsimData, dtaliteData] = await Promise.all([
        getRunSummary(RUN_ID),
        getRunEngines(RUN_ID),
        getEngineView(RUN_ID, "dlsim", 1500),
        getEngineView(RUN_ID, "dtalite", 1500).catch(() => null),
      ]);
      setSummary(summaryData);
      setEngineMeta(enginesData);
      setLeftView(dlsimData);
      setRightView(dtaliteData);
      setRefreshTick((n) => n + 1);
      setError((prev) => (prev && !runError ? "" : prev));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!sessionStarted) return;
    if (runError) return;
    loadRunData();
    const timer = setInterval(loadRunData, REFRESH_MS);
    return () => clearInterval(timer);
  }, [sessionStarted, runError]);

  async function onSubmitQuery(e) {
    e.preventDefault();
    if (!queryText.trim()) return;
    setRunningQuery(true);
    setRunError("");
    setError("");
    setSubmittedQuery(queryText.trim());
    setSummary((prev) => ({
      ...(prev || {}),
      query: queryText.trim(),
    }));
    setLeftView(null);
    setRightView(null);
    setEngineMeta(null);
    if (!sessionStarted) {
      setSessionStarted(true);
    }
    try {
      await runQuery(queryText.trim());
      setRunError("");
      await loadRunData();
    } catch (err) {
      setRunError(err.message);
      setError(err.message);
      setLeftView(null);
      setRightView(null);
    } finally {
      setRunningQuery(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <h1>AI-DLSIM Interactive Visualization</h1>
        <div className="toolbar">
          <span>Run: <strong>{RUN_ID}</strong></span>
          <span>Auto-refresh: {REFRESH_MS / 1000}s</span>
        </div>
      </header>

      <form className="query-form panel" onSubmit={onSubmitQuery}>
        <label htmlFor="query-input">Prompt</label>
        <input
          id="query-input"
          value={queryText}
          onChange={(e) => setQueryText(e.target.value)}
          placeholder="Type route query..."
        />
        <button type="submit" disabled={runningQuery}>
          {runningQuery ? "Running..." : "Run Query"}
        </button>
      </form>

      {error ? <div className="error-banner">{error}</div> : null}
      {loading ? <div className="loading">Loading run data...</div> : null}
      {!sessionStarted ? (
        <div className="panel welcome-banner">
          <h3>Welcome to AI-DLSIM</h3>
          <p>
            Explore trip simulation results on an interactive map. Enter a travel query above
            and click <strong>Run Query</strong> to generate a fresh route analysis.
          </p>
        </div>
      ) : null}

      {sessionStarted ? (
      <div className="compare-layout">
        <div className="left-stack">
          <div className="panel answer-panel">
            <h3>Latest Simulation Summary</h3>
            {(runError || runningQuery ? submittedQuery : summary?.query) ? (
              <p><strong>Prompt:</strong> {runError || runningQuery ? submittedQuery : summary?.query}</p>
            ) : null}

            <div className="quick-status">
              <span className={`pill pill-${displayStatus.toLowerCase().replace(" ", "-")}`}>
                {displayStatus}
              </span>
              <div className="quick-fields">
                <div>
                  <strong>Travel time:</strong>{" "}
                  {runningQuery
                    ? "Running..."
                    : runError
                      ? "N/A"
                      : leftRouteTT != null
                        ? `${leftRouteTT.toFixed(2)} min`
                        : "N/A"}
                </div>
                <div>
                  <strong>Route available:</strong>{" "}
                  {runningQuery ? "Running..." : runError ? "No" : routeAvailable ? "Yes" : "No"}
                </div>
                <div>
                  <strong>DLSim:</strong>{" "}
                  {runningQuery
                    ? "Running..."
                    : runError
                      ? "Query failed before simulation output"
                      : dlsimStatus}
                </div>
                <div>
                  <strong>DTALite:</strong>{" "}
                  {runningQuery ? "Running..." : runError ? "Not run" : dtaliteStatus}
                </div>
              </div>
            </div>

            <details className="details-block">
              <summary>Detailed LLM explanation</summary>
              <p className="answer-text">
                {runError
                  ? runError
                  : summary?.final_answer || "Run the query pipeline once to populate a human-readable final answer."}
              </p>
            </details>
          </div>

          <div className="dual-maps">
            <div className="panel pane">
              <h3>DLSim Route View</h3>
              <NetworkMap
                links={leftView?.network?.links || []}
                routePoints={leftView?.route?.route_points || []}
                mapKey={`left-${refreshTick}-${summary?.updated_at || "none"}`}
              />
            </div>
            <div className="panel pane">
              <h3>DTALite Traffic View</h3>
              <NetworkMap
                links={rightView?.network?.links || []}
                routePoints={rightView?.route?.route_points || []}
                mapKey={`right-${refreshTick}-${summary?.updated_at || "none"}`}
              />
            </div>
          </div>
        </div>

        <aside className="panel sidebar">
          <h3>Simulation Details</h3>
          <table className="cmp-table">
            <tbody>
              <tr>
                <td>Status</td>
                <td>{engineMeta?.engines?.dlsim?.exists ? "ready" : "missing"}</td>
                <td>{engineMeta?.engines?.dtalite?.exists ? "ready" : "missing"}</td>
              </tr>
              <tr>
                <td>Has agent.csv</td>
                <td>{engineMeta?.engines?.dlsim?.has_agent ? "yes" : "no"}</td>
                <td>{engineMeta?.engines?.dtalite?.has_agent ? "yes" : "no"}</td>
              </tr>
              <tr>
                <td>Has link_performance.csv</td>
                <td>{engineMeta?.engines?.dlsim?.has_link_performance ? "yes" : "no"}</td>
                <td>{engineMeta?.engines?.dtalite?.has_link_performance ? "yes" : "no"}</td>
              </tr>
              <tr>
                <td>Route travel time (min)</td>
                <td>{leftRouteTT ?? "-"}</td>
                <td>{rightRouteTT ?? "-"}</td>
              </tr>
              <tr>
                <td>Δ (DTALite - DLSim)</td>
                <td colSpan={2}>{ttDelta ?? "n/a"}</td>
              </tr>
            </tbody>
          </table>

          <div className="legend-block">
            <h4>Engine Roles</h4>
            <p><strong>DLSim:</strong> route/path output and baseline trip result.</p>
            <p><strong>DTALite:</strong> traffic evaluation on the route/network.</p>
          </div>

          <div className="legend-block">
            <h4>Output Paths</h4>
            <p className="mono">{summary?.engine_outputs?.dlsim_route_dir || "—"}</p>
            <p className="mono">{summary?.engine_outputs?.dtalite_traffic_dir || "—"}</p>
          </div>
        </aside>
      </div>
      ) : null}
    </div>
  );
}

