"""
Resolve place names to nearest network node IDs.

Flow: place name → geocode (lat/lon) → nearest node in node.csv via Haversine.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ResolvedLocation:
    place_name: str
    lat: float
    lon: float
    node_id: int
    distance_km: float


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_nodes(node_csv: Path) -> list[tuple[int, float, float]]:
    """Return list of (node_id, lat, lon) from node.csv."""
    nodes = []
    with open(node_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nid = row.get("node_id", "").strip()
            x = row.get("x_coord", "").strip()
            y = row.get("y_coord", "").strip()
            if nid and x and y:
                nodes.append((int(nid), float(y), float(x)))
    return nodes


def geocode_place(place_name: str) -> tuple[float, float]:
    """
    Geocode a place name to (lat, lon) using Nominatim.
    Falls back to OpenAI if Nominatim returns nothing.
    """
    import requests

    params = {
        "q": place_name,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": "AI-DLSIM/1.0 (traffic-simulation-research)"}
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        print(f"  [geocode warning] Nominatim failed for '{place_name}': {e}")

    raise ValueError(
        f"Could not geocode '{place_name}'. "
        "Try a more specific name (e.g., 'Cornell University, Ithaca, NY')."
    )


def find_nearest_node(
    lat: float, lon: float, nodes: list[tuple[int, float, float]]
) -> tuple[int, float]:
    """Return (node_id, distance_km) of the closest node."""
    best_id = -1
    best_dist = float("inf")
    for nid, nlat, nlon in nodes:
        d = _haversine_km(lat, lon, nlat, nlon)
        if d < best_dist:
            best_dist = d
            best_id = nid
    return best_id, best_dist


def resolve(place_name: str, node_csv: Path) -> ResolvedLocation:
    """Full pipeline: place name → geocode → nearest node."""
    lat, lon = geocode_place(place_name)
    nodes = _load_nodes(node_csv)
    if not nodes:
        raise ValueError(f"node.csv at {node_csv} is empty or has no valid rows.")
    node_id, dist_km = find_nearest_node(lat, lon, nodes)
    return ResolvedLocation(
        place_name=place_name,
        lat=lat,
        lon=lon,
        node_id=node_id,
        distance_km=round(dist_km, 4),
    )


def resolve_pair(
    origin: str, destination: str, node_csv: Path
) -> tuple[ResolvedLocation, ResolvedLocation]:
    """Resolve both origin and destination."""
    nodes = _load_nodes(node_csv)
    if not nodes:
        raise ValueError(f"node.csv at {node_csv} is empty or has no valid rows.")

    o_lat, o_lon = geocode_place(origin)
    o_id, o_dist = find_nearest_node(o_lat, o_lon, nodes)
    o_resolved = ResolvedLocation(origin, o_lat, o_lon, o_id, round(o_dist, 4))

    d_lat, d_lon = geocode_place(destination)
    d_id, d_dist = find_nearest_node(d_lat, d_lon, nodes)
    d_resolved = ResolvedLocation(destination, d_lat, d_lon, d_id, round(d_dist, 4))

    if o_id == d_id:
        raise ValueError(
            f"Origin and destination resolved to the same node ({o_id}). "
            "Try more specific place names."
        )

    return o_resolved, d_resolved
