from __future__ import annotations

import sys
from pathlib import Path

# Allow running as:
#   python src/ai_dlsim/workflows/run_baseline.py
SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from ai_dlsim.schemas.scenario import ScenarioPaths
from ai_dlsim.simulation.simulation_service import SimulationService, ensure_parent_dir


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    paths = ScenarioPaths(
        raw_osm_path=repo_root / "data" / "Ithaca" / "14850_roads.osm",
        node_csv_path=repo_root / "data" / "Ithaca" / "node.csv",
        link_csv_path=repo_root / "data" / "Ithaca" / "link.csv",
        demand_csv_path=repo_root / "data" / "Ithaca" / "demand.csv",
        output_dir=repo_root / "outputs" / "runs" / "ithaca_baseline",
    )

    ensure_parent_dir(paths.output_dir)
    result = SimulationService().run(paths)
    print(result)


if __name__ == "__main__":
    main()

