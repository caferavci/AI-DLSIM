const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function fetchJson(path) {
  const response = await fetch(`${API_BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function postJson(path, body) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    try {
      const data = await response.json();
      const detail = data?.detail;
      throw new Error(typeof detail === "string" ? detail : `API request failed: ${response.status} ${response.statusText}`);
    } catch {
      const text = await response.text();
      throw new Error(text || `API request failed: ${response.status} ${response.statusText}`);
    }
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

export function getRunEngines(runId) {
  return fetchJson(`/runs/${runId}/engines`);
}

export function getEngineView(runId, engine, maxLinks = 1500) {
  return fetchJson(`/runs/${runId}/engines/${engine}/view?max_links=${maxLinks}`);
}

export function runQuery(query, llmModel = "openai.gpt-5-mini") {
  return postJson("/query", { query, llm_model: llmModel });
}

