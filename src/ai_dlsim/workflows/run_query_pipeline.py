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
import shutil
import subprocess
import sys
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

DATA_DIR = REPO_ROOT / "data" / "Ithaca"
EXPECTED_OUTPUTS = ["link_performance.csv", "agent.csv", "solution.csv"]


def run_dlsim(run_dir: Path) -> dict:
    """Copy network files, run DLSim, return result metadata."""
    if not DLSIM_PY.exists():
        return {"status": "failed", "error": f"DLSim not found at {DLSIM_PY}. Run: git submodule update --init --recursive"}

    run_dir.mkdir(parents=True, exist_ok=True)

    for f in ["node.csv", "link.csv"]:
        shutil.copy(DATA_DIR / f, run_dir / f)

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


def parse_agent_results(run_dir: Path) -> dict:
    """Read agent.csv and extract travel time + route for the query agent."""
    agent_csv = run_dir / "agent.csv"
    if not agent_csv.exists():
        return {"travel_time_minutes": None, "route_nodes": None, "completed": False}

    with open(agent_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"travel_time_minutes": None, "route_nodes": None, "completed": False}

    agent = rows[0]
    travel_time = agent.get("travel_time", "")
    node_seq = agent.get("node_sequence", "")
    time_seq = agent.get("time_sequence", "")

    try:
        tt = float(travel_time) if travel_time else None
    except ValueError:
        tt = None

    return {
        "travel_time_minutes": tt,
        "route_nodes": node_seq or None,
        "time_sequence": time_seq or None,
        "completed": tt is not None and tt > 0,
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
    node_csv = DATA_DIR / "node.csv"
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

    # ── Step 4: Run DLSim
    print("\n[step 4] Running DLSim simulation ...")
    sim_result = run_dlsim(run_dir)

    if sim_result["status"] != "success":
        print(f"\n[error] Simulation failed: {sim_result.get('error', 'unknown')}")
        sys.exit(1)

    # ── Step 5: Parse outputs
    print("\n[step 5] Parsing simulation outputs ...")
    agent_result = parse_agent_results(run_dir)
    solution = parse_solution(run_dir)
    print(f"  travel_time: {agent_result.get('travel_time_minutes')} min")
    print(f"  route:       {agent_result.get('route_nodes')}")
    print(f"  completed:   {agent_result.get('completed')}")

    summary = {
        "query": args.query,
        "origin": {"name": origin_loc.place_name, "node_id": origin_loc.node_id},
        "destination": {"name": dest_loc.place_name, "node_id": dest_loc.node_id},
        "departure_time": request.departure_time,
        **agent_result,
        "solution": solution,
    }

    # ── Step 6: Post-processing agent (results → human answer)
    print("\n[step 6] Generating final answer with LLM ...")
    post = LlmResultInterpreter(model=args.llm_model)
    final_answer = post.interpret(user_query=args.query, dlsim_result=summary)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
