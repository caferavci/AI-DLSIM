"""
AI-DLSIM end-to-end query pipeline.

Usage:
    python3 src/ai_dlsim/workflows/run_query_pipeline.py \
        --query "What is the travel time from Cornell to Ithaca Commons at 8:30 AM?"

API keys are loaded automatically from .env in the repo root.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from ai_dlsim.preprocessing.llm_query_parser import LlmQueryParser
from ai_dlsim.preprocessing.location_resolver import resolve_pair
from ai_dlsim.preprocessing.input_agent_generator import generate_input_agent_csv
from ai_dlsim.postprocessing.llm_result_interpreter import LlmResultInterpreter

DLSIM_PY = REPO_ROOT / "external" / "DLSim" / "src" / "python" / "DLSim.py"
DTALITE_BIN = REPO_ROOT / "DTALite"

DATA_DIR = REPO_ROOT / "data" / "Ithaca"
FALLBACK_DATA_DIRS = [
    REPO_ROOT / "outputs" / "runs" / "query_pipeline",
    REPO_ROOT / "outputs" / "runs" / "ithaca_dlsim",
    REPO_ROOT / "data" / "14850",
]
EXPECTED_OUTPUTS = ["link_performance.csv", "agent.csv", "solution.csv"]


def normalize_link_lengths_for_dlsim(run_dir: Path) -> dict:
    """
    Normalize link.csv length units for DLSim runtime.

    Heuristic: if median(link.length) > 5, treat lengths as meters and
    convert to kilometers for the run copy only.
    """
    link_csv = run_dir / "link.csv"
    if not link_csv.exists():
        return {"status": "skipped", "reason": "link.csv missing"}

    with open(link_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = f.readline().strip().split(",") if not rows else list(rows[0].keys())

    if not rows:
        return {"status": "skipped", "reason": "link.csv empty"}

    lengths = []
    for row in rows:
        raw = (row.get("length") or "").strip()
        try:
            lengths.append(float(raw))
        except ValueError:
            continue

    if not lengths:
        return {"status": "skipped", "reason": "no numeric length values"}

    median_length = statistics.median(lengths)
    if median_length <= 5:
        return {
            "status": "unchanged",
            "assumed_unit": "km_or_miles",
            "median_length": median_length,
        }

    converted = 0
    for row in rows:
        raw = (row.get("length") or "").strip()
        try:
            km = float(raw) / 1000.0
            row["length"] = f"{km:.6f}"
            converted += 1
        except ValueError:
            continue

    with open(link_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "status": "converted",
        "assumed_unit": "meters",
        "median_length": median_length,
        "rows_converted": converted,
    }


def resolve_network_source_dir() -> Path:
    """Pick first directory that contains both node.csv and link.csv."""
    candidates = [DATA_DIR, *FALLBACK_DATA_DIRS]
    for directory in candidates:
        if (directory / "node.csv").exists() and (directory / "link.csv").exists():
            return directory
    raise FileNotFoundError(
        "Could not find node.csv/link.csv in expected sources: "
        + ", ".join(str(d) for d in candidates)
    )


def run_dlsim(run_dir: Path) -> dict:
    """Copy network files, run DLSim, return result metadata."""
    if not DLSIM_PY.exists():
        return {"status": "failed", "error": f"DLSim not found at {DLSIM_PY}. Run: git submodule update --init --recursive"}

    run_dir.mkdir(parents=True, exist_ok=True)

    for f in ["node.csv", "link.csv"]:
        shutil.copy(DATA_DIR / f, run_dir / f)

    normalization = normalize_link_lengths_for_dlsim(run_dir)
    if normalization["status"] == "converted":
        print(
            "[sim] Normalized link lengths for runtime copy: "
            f"assumed meters (median={normalization['median_length']:.3f})"
        )
    elif normalization["status"] == "unchanged":
        print(
            "[sim] Kept link lengths unchanged: "
            f"median={normalization['median_length']:.3f}"
        )
    else:
        print(f"[sim] Link length normalization skipped: {normalization.get('reason', 'unknown')}")

    print(f"\n[sim] Running DLSim from {run_dir} ...")
    result = subprocess.run(
        [sys.executable, str(DLSIM_PY)],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {"status": "failed", "error": result.stderr or result.stdout, "output_dir": str(run_dir)}

    produced = [f for f in EXPECTED_OUTPUTS if (run_dir / f).exists()]
    return {"status": "success", "output_dir": str(run_dir), "produced": produced}


def run_dlsim_route_engine(base_run_dir: Path) -> dict:
    """Run DLSim route engine in an isolated folder."""
    engine_dir = base_run_dir / "dlsim_engine"
    engine_dir.mkdir(parents=True, exist_ok=True)

    # Preserve input_agent generated in base_run_dir.
    input_agent = base_run_dir / "input_agent.csv"
    if not input_agent.exists():
        return {"status": "failed", "error": f"Missing {input_agent}"}

    source_dir = resolve_network_source_dir()
    for f in ["node.csv", "link.csv"]:
        shutil.copy(source_dir / f, engine_dir / f)
    shutil.copy(input_agent, engine_dir / "input_agent.csv")

    normalization = normalize_link_lengths_for_dlsim(engine_dir)
    if normalization["status"] == "converted":
        print(
            "[route-engine] Normalized link lengths: "
            f"assumed meters (median={normalization['median_length']:.3f})"
        )

    print(f"\n[route-engine] Running DLSim from {engine_dir} ...")
    result = subprocess.run(
        [sys.executable, str(DLSIM_PY)],
        cwd=str(engine_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return {
            "status": "failed",
            "error": result.stderr or result.stdout,
            "output_dir": str(engine_dir),
        }

    produced = [f for f in EXPECTED_OUTPUTS if (engine_dir / f).exists()]
    return {"status": "success", "output_dir": str(engine_dir), "produced": produced}


def run_dtalite_traffic_engine(base_run_dir: Path) -> dict:
    """Run DTALite traffic engine in an isolated folder."""
    engine_dir = base_run_dir / "dtalite_engine"
    engine_dir.mkdir(parents=True, exist_ok=True)

    if not DTALITE_BIN.exists():
        return {
            "status": "failed",
            "error": f"DTALite binary not found at {DTALITE_BIN}",
            "output_dir": str(engine_dir),
        }

    input_agent = base_run_dir / "input_agent.csv"
    if not input_agent.exists():
        return {"status": "failed", "error": f"Missing {input_agent}", "output_dir": str(engine_dir)}

    source_dir = resolve_network_source_dir()
    for f in ["node.csv", "link.csv"]:
        shutil.copy(source_dir / f, engine_dir / f)
    shutil.copy(input_agent, engine_dir / "input_agent.csv")

    normalization = normalize_link_lengths_for_dlsim(engine_dir)
    if normalization["status"] == "converted":
        print(
            "[traffic-engine] Normalized link lengths: "
            f"assumed meters (median={normalization['median_length']:.3f})"
        )

    print(f"\n[traffic-engine] Running DTALite from {engine_dir} ...")
    try:
        result = subprocess.run(
            [str(DTALITE_BIN)],
            cwd=str(engine_dir),
            capture_output=True,
            text=True,
        )
    except OSError as e:
        return {
            "status": "failed",
            "error": f"DTALite execution failed: {e}",
            "output_dir": str(engine_dir),
        }

    if result.returncode != 0:
        return {
            "status": "failed",
            "error": result.stderr or result.stdout,
            "output_dir": str(engine_dir),
        }

    produced = [f for f in EXPECTED_OUTPUTS if (engine_dir / f).exists()]
    return {"status": "success", "output_dir": str(engine_dir), "produced": produced}


def parse_agent_results(run_dir: Path) -> dict:
    """Read agent.csv and extract travel time + route for the query agent."""
    agent_csv = run_dir / "agent.csv"
    if not agent_csv.exists():
        return {
            "travel_time_minutes": None,
            "route_nodes": None,
            "completed": False,
            "status": "missing_agent_csv",
        }

    with open(agent_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {
            "travel_time_minutes": None,
            "route_nodes": None,
            "completed": False,
            "status": "no_agent_rows",
        }

    agent = rows[0]
    travel_time = agent.get("travel_time", "")
    node_seq = agent.get("node_sequence", "")
    time_seq = agent.get("time_sequence", "")

    try:
        tt = float(travel_time) if travel_time else None
    except ValueError:
        tt = None

    # DLSim python output can report negative travel_time even when
    # time_sequence is valid. Use timestamps as a fallback in that case.
    if (tt is None or tt <= 0) and time_seq:
        times = [t.strip() for t in time_seq.split(";") if t.strip()]
        if len(times) >= 2 and all("-" not in t for t in times):
            try:
                t0 = datetime.strptime(times[0], "%H%M:%S")
                t1 = datetime.strptime(times[-1], "%H%M:%S")
                fallback_minutes = (t1 - t0).total_seconds() / 60.0
                if fallback_minutes < 0:
                    fallback_minutes += 24 * 60
                if fallback_minutes > 0:
                    tt = fallback_minutes
            except ValueError:
                pass

    return {
        "travel_time_minutes": tt,
        "route_nodes": node_seq or None,
        "time_sequence": time_seq or None,
        "completed": tt is not None and tt > 0,
        "status": "ok" if tt is not None and tt > 0 else "invalid_or_incomplete",
    }


def parse_solution(run_dir: Path) -> dict:
    """Read solution.csv for run summary stats."""
    sol_csv = run_dir / "solution.csv"
    if not sol_csv.exists():
        return {}
    with open(sol_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return dict(rows[0])


def _route_pairs(node_sequence: str | None) -> list[tuple[str, str]]:
    nodes = [n for n in (node_sequence or "").split(";") if n]
    return list(zip(nodes, nodes[1:]))


def parse_dtalite_route_traffic(run_dir: Path, route_nodes: str | None) -> dict:
    """Extract DTALite link-level traffic values along DLSim route."""
    lp_csv = run_dir / "link_performance.csv"
    if not lp_csv.exists():
        return {"status": "missing_output", "route_link_metrics": []}

    route_edges = set(_route_pairs(route_nodes))
    if not route_edges:
        return {"status": "no_route_edges", "route_link_metrics": []}

    route_metrics: list[dict] = []
    with open(lp_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            from_id = (row.get("from_node_id") or "").strip()
            to_id = (row.get("to_node_id") or "").strip()
            if (from_id, to_id) not in route_edges:
                continue
            route_metrics.append(
                {
                    "from_node_id": from_id,
                    "to_node_id": to_id,
                    "time_period": row.get("time_period"),
                    "speed": row.get("speed"),
                    "volume": row.get("volume"),
                    "travel_time": row.get("travel_time"),
                    "queue": row.get("queue"),
                }
            )

    return {
        "status": "success",
        "route_link_metrics": route_metrics,
        "count": len(route_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-DLSIM end-to-end query pipeline.")
    parser.add_argument("--query", type=str, required=True, help="Natural language query.")
    parser.add_argument(
        "--llm-model", type=str, default="openai.gpt-5-mini",
        help="LLM model for pre/post agents.",
    )
    args = parser.parse_args()

    # ── Step 1: Pre-processing agent (NL → structured params)
    print("\n[step 1] Parsing query with LLM ...")
    pre = LlmQueryParser(model=args.llm_model)
    request = pre.parse(args.query)
    print(f"  region:         {request.region}")
    print(f"  origin:         {request.origin}")
    print(f"  destination:    {request.destination}")
    print(f"  departure_time: {request.departure_time}")
    print(f"  mode:           {request.mode}")

    if not request.origin or not request.destination:
        print("\n[error] Query must specify both origin and destination.")
        sys.exit(1)

    # ── Step 2: Resolve place names → node IDs
    print("\n[step 2] Resolving locations to network nodes ...")
    source_dir = resolve_network_source_dir()
    print(f"  network source: {source_dir}")
    node_csv = source_dir / "node.csv"
    origin_loc, dest_loc = resolve_pair(request.origin, request.destination, node_csv)
    print(f"  origin:      '{origin_loc.place_name}' → node {origin_loc.node_id} ({origin_loc.distance_km} km away)")
    print(f"  destination: '{dest_loc.place_name}' → node {dest_loc.node_id} ({dest_loc.distance_km} km away)")

    # ── Step 3: Generate input_agent.csv
    run_dir = REPO_ROOT / "outputs" / "runs" / "query_pipeline"
    agent_csv_path = run_dir / "input_agent.csv"
    generate_input_agent_csv(
        o_node_id=origin_loc.node_id,
        d_node_id=dest_loc.node_id,
        departure_time_str=request.departure_time,
        output_path=agent_csv_path,
    )
    print(f"\n[step 3] Generated {agent_csv_path}")

    # ── Step 4a: Route engine (DLSim)
    print("\n[step 4a] Running route engine (DLSim) ...")
    route_engine_result = run_dlsim_route_engine(run_dir)
    if route_engine_result["status"] != "success":
        print(f"\n[error] Route engine failed: {route_engine_result.get('error', 'unknown')}")
        sys.exit(1)

    dlsim_dir = Path(route_engine_result["output_dir"])
    # Copy route-engine outputs to top-level query_pipeline for compatibility
    for f in ["agent.csv", "link_performance.csv", "solution.csv", "node.csv", "link.csv"]:
        src = dlsim_dir / f
        if src.exists():
            shutil.copy(src, run_dir / f)

    # ── Step 4b: Traffic engine (DTALite)
    print("\n[step 4b] Running traffic engine (DTALite) ...")
    traffic_engine_result = run_dtalite_traffic_engine(run_dir)
    if traffic_engine_result["status"] != "success":
        print(f"  [warn] Traffic engine failed: {traffic_engine_result.get('error', 'unknown')}")

    # ── Step 5: Parse route engine outputs
    print("\n[step 5] Parsing route engine outputs ...")
    agent_result = parse_agent_results(dlsim_dir)
    solution = parse_solution(dlsim_dir)
    print(f"  travel_time: {agent_result.get('travel_time_minutes')} min")
    print(f"  route:       {agent_result.get('route_nodes')}")
    print(f"  completed:   {agent_result.get('completed')}")

    # ── Step 5b: Parse traffic engine outputs (on selected route)
    print("\n[step 5b] Parsing traffic engine outputs ...")
    traffic_result = {"status": "not_run", "route_link_metrics": []}
    if traffic_engine_result["status"] == "success":
        dtalite_dir = Path(traffic_engine_result["output_dir"])
        traffic_result = parse_dtalite_route_traffic(dtalite_dir, agent_result.get("route_nodes"))
        print(f"  route link metrics found: {traffic_result.get('count', 0)}")
    else:
        print("  route link metrics found: 0 (traffic engine unavailable)")

    summary = {
        "query": args.query,
        "origin": {"name": origin_loc.place_name, "node_id": origin_loc.node_id},
        "destination": {"name": dest_loc.place_name, "node_id": dest_loc.node_id},
        "departure_time": request.departure_time,
        "engines": {
            "dlsim_route_engine": route_engine_result,
            "dtalite_traffic_engine": traffic_engine_result,
        },
        "traffic_on_route": traffic_result,
        **agent_result,
        "solution": solution,
    }

    # ── Step 6: Post-processing agent (results → human answer)
    print("\n[step 6] Generating final answer with LLM ...")
    if not agent_result.get("completed"):
        dlsim_issue = agent_result.get("status", "unknown")
        dtalite_issue = traffic_engine_result.get("error", "not available")
        final_answer = (
            "Travel time: N/A\n"
            f"Route available: {'No' if dlsim_issue in {'no_agent_rows', 'missing_agent_csv'} else 'Unknown'}\n"
            f"DLSim: {'No feasible agent output' if dlsim_issue in {'no_agent_rows', 'missing_agent_csv'} else dlsim_issue}\n"
            f"DTALite: {'Binary incompatible' if 'Bad CPU type' in dtalite_issue else 'Unavailable'}\n"
            "\n"
            "Try a nearby landmark pair or a known-good OD (e.g., Cornell University ↔ Ithaca Commons)."
        )
    else:
        post = LlmResultInterpreter(model=args.llm_model)
        final_answer = post.interpret(user_query=args.query, dlsim_result=summary)

    # Persist latest text/structured results for dashboard consumption.
    (run_dir / "final_answer.txt").write_text(final_answer, encoding="utf-8")
    (run_dir / "dashboard_summary.json").write_text(
        json.dumps(
            {
                "query": args.query,
                "final_answer": final_answer,
                "travel_time_minutes": agent_result.get("travel_time_minutes"),
                "completed": agent_result.get("completed"),
                "origin": summary["origin"],
                "destination": summary["destination"],
                "departure_time": summary["departure_time"],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "engine_outputs": {
                    "dlsim_route_dir": str(Path(route_engine_result["output_dir"])),
                    "dtalite_traffic_dir": str(Path(traffic_engine_result.get("output_dir", ""))),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
