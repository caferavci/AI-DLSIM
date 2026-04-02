"""
Generate input_agent.csv for DLSim from resolved origin/destination node IDs.
"""
from __future__ import annotations

import csv
from pathlib import Path


def time_str_to_minutes(time_str: str | None) -> int:
    """Convert 'HH:MM' to minutes from midnight. Default 08:00 if None."""
    if not time_str:
        return 480
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        return 480
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except ValueError:
        return 480


def generate_input_agent_csv(
    *,
    o_node_id: int,
    d_node_id: int,
    departure_time_str: str | None,
    output_path: Path,
    pce: float = 1.0,
    path_fixed_flag: int = 0,
) -> Path:
    """
    Write a single-agent input_agent.csv that DLSim can consume directly.

    For v1, we generate one agent per query. Later this can be extended
    to support multiple agents or integrate with grid2demand output.
    """
    departure_min = time_str_to_minutes(departure_time_str)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "o_node_id",
        "d_node_id",
        "departure_time_in_min",
        "PCE",
        "path_fixed_flag",
        "path_node_sequence",
    ]

    row = {
        "o_node_id": o_node_id,
        "d_node_id": d_node_id,
        "departure_time_in_min": departure_min,
        "PCE": pce,
        "path_fixed_flag": path_fixed_flag,
        "path_node_sequence": "",
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    return output_path
