from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScenarioPaths:
    """File paths expected/produced by the deterministic pipeline."""

    raw_osm_path: Path
    node_csv_path: Path
    link_csv_path: Path
    demand_csv_path: Path
    output_dir: Path

