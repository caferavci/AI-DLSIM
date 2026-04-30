from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "outputs" / "runs"
QUERY_PIPELINE_PY = REPO_ROOT / "src" / "ai_dlsim" / "workflows" / "run_query_pipeline.py"
ENGINE_FOLDERS = {
    "dlsim": "dlsim_engine",
    "dtalite": "dtalite_engine",
}

app = FastAPI(title="AI-DLSIM Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRunRequest(BaseModel):
    query: str = Field(min_length=5, max_length=500)
    llm_model: str = Field(default="openai.gpt-5-mini", min_length=3, max_length=100)


def _run_dir(run_id: str) -> Path:
    path = RUNS_DIR / run_id
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return path


def _engine_dir(run_id: str, engine: str) -> Path:
    run_dir = _run_dir(run_id)
    folder = ENGINE_FOLDERS.get(engine.lower())
    if not folder:
        raise HTTPException(status_code=400, detail=f"Unknown engine '{engine}'.")
    path = run_dir / folder
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Engine folder not found for '{engine}'.")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_linestring_wkt(wkt: str) -> list[list[float]]:
    text = (wkt or "").strip()
    if not text.startswith("LINESTRING"):
        return []
    start = text.find("(")
    end = text.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return []

    payload = text[start + 1 : end]
    coords: list[list[float]] = []
    for part in payload.split(","):
        tokens = part.strip().split()
        if len(tokens) < 2:
            continue
        try:
            lon = float(tokens[0])
            lat = float(tokens[1])
            coords.append([lat, lon])
        except ValueError:
            continue
    return coords


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def _compute_travel_time_from_time_sequence(time_sequence: str) -> float | None:
    times = [t.strip() for t in (time_sequence or "").split(";") if t.strip()]
    if len(times) < 2 or any("-" in t for t in times):
        return None
    try:
        t0 = datetime.strptime(times[0], "%H%M:%S")
        t1 = datetime.strptime(times[-1], "%H%M:%S")
    except ValueError:
        return None
    minutes = (t1 - t0).total_seconds() / 60.0
    if minutes < 0:
        minutes += 24 * 60
    return minutes if minutes > 0 else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def run_query_from_ui(payload: QueryRunRequest) -> dict:
    if not QUERY_PIPELINE_PY.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline script not found at {QUERY_PIPELINE_PY}",
        )

    cmd = [
        sys.executable,
        str(QUERY_PIPELINE_PY),
        "--query",
        payload.query,
        "--llm-model",
        payload.llm_model,
    ]
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-3000:]
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline run failed.\n{tail}",
        )

    summary = run_summary("query_pipeline")
    return {
        "status": "success",
        "run_id": "query_pipeline",
        "summary": summary,
    }


@app.get("/runs")
def list_runs() -> dict[str, list[str]]:
    if not RUNS_DIR.exists():
        return {"runs": []}
    runs = [p.name for p in RUNS_DIR.iterdir() if p.is_dir()]
    runs.sort()
    return {"runs": runs}


@app.get("/runs/{run_id}/summary")
def run_summary(run_id: str) -> dict:
    run_dir = _run_dir(run_id)
    solution_rows = _read_csv(run_dir / "solution.csv")
    agent_rows = _read_csv(run_dir / "agent.csv")
    dashboard_summary_path = run_dir / "dashboard_summary.json"

    solution = solution_rows[0] if solution_rows else {}
    agent = agent_rows[0] if agent_rows else {}
    time_sequence = agent.get("time_sequence", "")
    raw_tt = _to_float(agent.get("travel_time"))
    derived_tt = _compute_travel_time_from_time_sequence(time_sequence)
    travel_time = raw_tt if raw_tt is not None and raw_tt > 0 else derived_tt
    completed = travel_time is not None and travel_time > 0

    final_answer = ""
    query_text = ""
    updated_at = ""
    engine_outputs: dict[str, str] = {}
    if dashboard_summary_path.exists():
        try:
            parsed = json.loads(dashboard_summary_path.read_text(encoding="utf-8"))
            final_answer = parsed.get("final_answer", "")
            query_text = parsed.get("query", "")
            updated_at = parsed.get("updated_at", "")
            engine_outputs = parsed.get("engine_outputs", {})
        except json.JSONDecodeError:
            pass

    return {
        "run_id": run_id,
        "solution": solution,
        "query": query_text,
        "final_answer": final_answer,
        "completed": completed,
        "updated_at": updated_at,
        "engine_outputs": engine_outputs,
        "agent": {
            "agent_id": agent.get("agent_id"),
            "o_node_id": agent.get("o_node_id"),
            "d_node_id": agent.get("d_node_id"),
            "travel_time": travel_time,
            "raw_travel_time": raw_tt,
            "node_sequence": agent.get("node_sequence", ""),
            "time_sequence": time_sequence,
        },
    }


@app.get("/runs/{run_id}/network")
def run_network(
    run_id: str,
    max_links: int = Query(default=1500, ge=1, le=10000),
) -> dict:
    run_dir = _run_dir(run_id)
    links = _read_csv(run_dir / "link.csv")
    link_perf = _read_csv(run_dir / "link_performance.csv")

    perf_by_link: dict[str, dict[str, str]] = {}
    for row in link_perf:
        link_id = row.get("link_id")
        if link_id and link_id not in perf_by_link:
            perf_by_link[link_id] = row

    features = []
    for row in links[:max_links]:
        link_id = row.get("link_id", "")
        perf = perf_by_link.get(link_id, {})
        features.append(
            {
                "link_id": link_id,
                "from_node_id": row.get("from_node_id"),
                "to_node_id": row.get("to_node_id"),
                "geometry": _parse_linestring_wkt(row.get("geometry", "")),
                "speed": _to_float(perf.get("speed")),
                "volume": _to_float(perf.get("volume")),
                "travel_time": _to_float(perf.get("travel_time")),
            }
        )

    return {"run_id": run_id, "link_count": len(features), "links": features}


@app.get("/runs/{run_id}/route")
def run_route(run_id: str) -> dict:
    run_dir = _run_dir(run_id)
    agent_rows = _read_csv(run_dir / "agent.csv")
    node_rows = _read_csv(run_dir / "node.csv")

    if not agent_rows:
        raise HTTPException(status_code=404, detail="agent.csv not found or empty.")

    node_coord: dict[str, list[float]] = {}
    for row in node_rows:
        node_id = row.get("node_id")
        if not node_id:
            continue
        lat = _to_float(row.get("y_coord"))
        lon = _to_float(row.get("x_coord"))
        if lat is None or lon is None:
            continue
        node_coord[node_id] = [lat, lon]

    agent = agent_rows[0]
    node_seq = [n for n in (agent.get("node_sequence") or "").split(";") if n]
    time_seq = [t for t in (agent.get("time_sequence") or "").split(";") if t]
    route_points = [node_coord[n] for n in node_seq if n in node_coord]

    return {
        "run_id": run_id,
        "agent_id": agent.get("agent_id"),
        "travel_time": _to_float(agent.get("travel_time")),
        "node_sequence": node_seq,
        "time_sequence": time_seq,
        "route_points": route_points,
    }


def _engine_network(engine_dir: Path, max_links: int) -> dict:
    links = _read_csv(engine_dir / "link.csv")
    link_perf = _read_csv(engine_dir / "link_performance.csv")

    perf_by_link: dict[str, dict[str, str]] = {}
    for row in link_perf:
        link_id = row.get("link_id")
        if link_id and link_id not in perf_by_link:
            perf_by_link[link_id] = row

    features = []
    for row in links[:max_links]:
        link_id = row.get("link_id", "")
        perf = perf_by_link.get(link_id, {})
        features.append(
            {
                "link_id": link_id,
                "from_node_id": row.get("from_node_id"),
                "to_node_id": row.get("to_node_id"),
                "geometry": _parse_linestring_wkt(row.get("geometry", "")),
                "speed": _to_float(perf.get("speed")),
                "volume": _to_float(perf.get("volume")),
                "travel_time": _to_float(perf.get("travel_time")),
            }
        )
    return {"link_count": len(features), "links": features}


def _engine_route(engine_dir: Path) -> dict:
    agent_rows = _read_csv(engine_dir / "agent.csv")
    node_rows = _read_csv(engine_dir / "node.csv")
    if not agent_rows:
        return {
            "available": False,
            "reason": "agent.csv not found",
            "route_points": [],
            "node_sequence": [],
            "time_sequence": [],
            "travel_time": None,
        }

    node_coord: dict[str, list[float]] = {}
    for row in node_rows:
        node_id = row.get("node_id")
        if not node_id:
            continue
        lat = _to_float(row.get("y_coord"))
        lon = _to_float(row.get("x_coord"))
        if lat is None or lon is None:
            continue
        node_coord[node_id] = [lat, lon]

    agent = agent_rows[0]
    node_seq = [n for n in (agent.get("node_sequence") or "").split(";") if n]
    time_seq = [t for t in (agent.get("time_sequence") or "").split(";") if t]
    route_points = [node_coord[n] for n in node_seq if n in node_coord]
    tt = _to_float(agent.get("travel_time"))
    if tt is None or tt <= 0:
        tt = _compute_travel_time_from_time_sequence(agent.get("time_sequence", ""))

    return {
        "available": True,
        "agent_id": agent.get("agent_id"),
        "travel_time": tt,
        "node_sequence": node_seq,
        "time_sequence": time_seq,
        "route_points": route_points,
    }


@app.get("/runs/{run_id}/engines")
def run_engines(run_id: str) -> dict:
    run_dir = _run_dir(run_id)
    data = {}
    for key, folder in ENGINE_FOLDERS.items():
        path = run_dir / folder
        data[key] = {
            "exists": path.exists() and path.is_dir(),
            "path": str(path),
            "has_link_performance": (path / "link_performance.csv").exists(),
            "has_agent": (path / "agent.csv").exists(),
            "has_solution": (path / "solution.csv").exists(),
        }
    return {"run_id": run_id, "engines": data}


@app.get("/runs/{run_id}/engines/{engine}/view")
def run_engine_view(
    run_id: str,
    engine: str,
    max_links: int = Query(default=1500, ge=1, le=10000),
) -> dict:
    if not isinstance(max_links, int):
        max_links = 1500
    engine_dir = _engine_dir(run_id, engine)
    solution_rows = _read_csv(engine_dir / "solution.csv")
    solution = solution_rows[0] if solution_rows else {}
    network = _engine_network(engine_dir, max_links)
    route = _engine_route(engine_dir)
    return {
        "run_id": run_id,
        "engine": engine.lower(),
        "solution": solution,
        "network": network,
        "route": route,
    }

