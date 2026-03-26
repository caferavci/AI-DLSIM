from __future__ import annotations

from pathlib import Path


class DlsimAdapter:
    """
    Thin boundary to the simulation engine.

    Baseline stub: later this will be replaced to call external/DLSim.
    """

    def run(
        self,
        *,
        node_csv: Path,
        link_csv: Path,
        demand_csv: Path,
        output_dir: Path,
        simulation_horizon_seconds: int = 3600,
        time_step_seconds: int = 1,
        capacity_multiplier: float = 1.0,
    ) -> dict:
        # Baseline stub intentionally does not touch the filesystem
        # beyond returning a structured placeholder result.
        return {
            "engine": "DLSim",
            "status": "not_implemented",
            "inputs": {
                "node_csv": str(node_csv),
                "link_csv": str(link_csv),
                "demand_csv": str(demand_csv),
            },
            "output_dir": str(output_dir),
            "simulation_settings": {
                "simulation_horizon_seconds": simulation_horizon_seconds,
                "time_step_seconds": time_step_seconds,
                "capacity_multiplier": capacity_multiplier,
            },
        }

