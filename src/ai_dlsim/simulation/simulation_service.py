from __future__ import annotations

from pathlib import Path

from ai_dlsim.adapters.dlsim_adapter import DlsimAdapter
from ai_dlsim.schemas.scenario import ScenarioPaths


class SimulationService:
    def __init__(self, adapter: DlsimAdapter | None = None) -> None:
        self.adapter = adapter or DlsimAdapter()

    def run(
        self,
        paths: ScenarioPaths,
        *,
        simulation_horizon_seconds: int = 3600,
        time_step_seconds: int = 1,
        capacity_multiplier: float = 1.0,
    ) -> dict:
        # Note: Phase 4 will extend this to actually invoke external/DLSim.
        # For baseline, the adapter returns a structured placeholder.
        return self.adapter.run(
            node_csv=paths.node_csv_path,
            link_csv=paths.link_csv_path,
            demand_csv=paths.demand_csv_path,
            output_dir=paths.output_dir,
            simulation_horizon_seconds=simulation_horizon_seconds,
            time_step_seconds=time_step_seconds,
            capacity_multiplier=capacity_multiplier,
        )


def ensure_parent_dir(path: Path) -> None:
    """Utility used by later phases to create output folders."""

    path.parent.mkdir(parents=True, exist_ok=True)

