"""
Run DLSim for a given scenario.

Usage:
    python3 scripts/run_dlsim.py

This script:
  1. Copies node.csv, link.csv, input_agent.csv into a clean run directory.
  2. Invokes DLSim.py from that directory (via the external/DLSim submodule).
  3. Reports outputs (link_performance.csv, agent.csv, solution.csv).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DLSIM_PY  = REPO_ROOT / "external" / "DLSim" / "src" / "python" / "DLSim.py"

DATA_DIR   = REPO_ROOT / "data" / "Ithaca"
OUTPUT_DIR = REPO_ROOT / "outputs" / "runs" / "ithaca_dlsim"

REQUIRED_INPUTS  = ["node.csv", "link.csv", "input_agent.csv"]
EXPECTED_OUTPUTS = ["link_performance.csv", "agent.csv", "solution.csv"]


def resolve_dlsim() -> Path:
    if DLSIM_PY.exists():
        return DLSIM_PY
    print("[error] DLSim submodule not initialised.")
    print(f"  Expected: {DLSIM_PY}")
    print("  Run:  git submodule update --init --recursive")
    sys.exit(1)


def validate_inputs(data_dir: Path) -> None:
    missing = [f for f in REQUIRED_INPUTS if not (data_dir / f).exists()]
    if missing:
        print(f"[error] Missing required input files in {data_dir}:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)


def setup_run_dir(data_dir: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for fname in REQUIRED_INPUTS:
        shutil.copy(data_dir / fname, run_dir / fname)
    print(f"[info] Run directory ready: {run_dir}")


def run(data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR) -> dict:
    dlsim_py = resolve_dlsim()
    validate_inputs(data_dir)
    setup_run_dir(data_dir, output_dir)

    print(f"[info] Running DLSim from {output_dir} ...")
    result = subprocess.run(
        [sys.executable, str(dlsim_py)],
        cwd=str(output_dir),
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"[error] DLSim exited with code {result.returncode}")
        return {"status": "failed", "returncode": result.returncode, "output_dir": str(output_dir)}

    produced = [f for f in EXPECTED_OUTPUTS if (output_dir / f).exists()]
    missing  = [f for f in EXPECTED_OUTPUTS if f not in produced]

    print("\n[info] Output files:")
    for f in produced:
        size = (output_dir / f).stat().st_size
        print(f"  {f}  ({size:,} bytes)")
    if missing:
        print(f"[warn] Missing expected outputs: {missing}")

    return {
        "status": "success",
        "returncode": 0,
        "output_dir": str(output_dir),
        "produced": produced,
        "missing": missing,
    }


if __name__ == "__main__":
    summary = run()
    print("\n[summary]", summary)
