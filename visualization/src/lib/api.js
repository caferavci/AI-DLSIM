const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

export function getRuns() {
  return fetchJson("/runs");
}

export function getRunSummary(runId) {
  return fetchJson(`/runs/${runId}/summary`);
}

export function getRunNetwork(runId, maxLinks = 1500) {
  return fetchJson(`/runs/${runId}/network?max_links=${maxLinks}`);
}

export function getRunRoute(runId) {
  return fetchJson(`/runs/${runId}/route`);
}

