"""
generate_settings.py

Generates settings.csv for DLSim-MRM from a preprocessed data folder.

Reads link.csv to discover which link types are present, then writes settings.csv
into the sibling <folder>_demand directory (where demand.csv from run_grid2demand.py
lives).  Also writes node.csv with zone_ids aligned to grid2demand's zones, and a
DLSim-compatible zone.csv.

Usage:
    python generate_settings.py
    python generate_settings.py data/14850
    python generate_settings.py data/14850 --period PM --time 1500_1800
"""

import sys
import csv
import shutil
import pathlib

csv.field_size_limit(sys.maxsize)

# Mapping: osm2gmns link_type integer -> (link_type_name, type_code)
# type_code:  f = freeway,  a = arterial,  r = ramp,  c = centroid connector
LINK_TYPE_CODES: dict[int, tuple[str, str]] = {
    1:  ("motorway",        "f"),
    2:  ("trunk",           "f"),
    3:  ("primary",         "a"),
    4:  ("secondary",       "a"),
    5:  ("tertiary",        "a"),
    6:  ("residential",     "a"),
    7:  ("ramp",            "r"),
    8:  ("service",         "a"),
    9:  ("connector",       "c"),
    11: ("motorway_link",   "f"),
    12: ("trunk_link",      "f"),
    13: ("primary_link",    "a"),
    14: ("secondary_link",  "a"),
    15: ("tertiary_link",   "a"),
    20: ("unclassified",    "a"),
}


def read_link_types(link_csv: pathlib.Path) -> list:
    """Return sorted list of (link_type_id, name, type_code) found in link.csv."""
    seen = {}
    with open(link_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("link_type", "").strip()
            if not raw or not raw.isdigit():
                continue
            lt = int(raw)
            if lt not in seen:
                seen[lt] = row.get("facility_type", row.get("link_type_name", "")).strip()

    result = []
    for lt, name_from_csv in sorted(seen.items()):
        fallback_name, type_code = LINK_TYPE_CODES.get(lt, (str(lt), "a"))
        name = name_from_csv or fallback_name
        result.append((lt, name, type_code))
    return result


def write_settings_csv(out_path, link_types, demand_period, time_period):
    rows = []

    # [assignment]
    rows.append([
        "[assignment]", "assignment_mode", "column_generation_iterations",
        "column_updating_iterations", "odme_iterations",
        "simulation_iterations", "number_of_memory_blocks",
    ])
    rows.append(["", "dta", "20", "20", "400", "0", "4"])
    rows.append([])

    # [agent_type]
    rows.append(["[agent_type]", "agent_type", "name", "", "vot", "flow_type", "pce", "occ"])
    rows.append(["", "auto", "passenger", "", "10", "0", "1", "1"])
    rows.append([])

    # [link_type]
    rows.append([
        "[link_type]", "link_type", "link_type_name", "",
        "agent_type_blocklist", "type_code", "traffic_flow_code", "vdf_type",
    ])
    for lt, name, type_code in link_types:
        rows.append(["", str(lt), name, "", "", type_code, "1", "qvdf"])
    rows.append([])

    # [demand_period]
    rows.append(["[demand_period]", "demand_period_id", "demand_period", "", "time_period"])
    rows.append(["", "1", demand_period, "", time_period])
    rows.append([])

    # [demand_file_list]
    rows.append([
        "[demand_file_list]", "file_sequence_no", "file_name", "",
        "format_type", "demand_period", "agent_type", "loading_scale_factor",
    ])
    rows.append(["", "1", "demand.csv", "", "column", demand_period, "auto", "1"])
    rows.append([])

    # [output_file_configuration]
    rows.append([
        "[output_file_configuration]", "", "path_output",
        "major_path_volume_threshold", "trajectory_output",
        "trajectory_sampling_rate", "trajectory_diversion_only", "",
        "td_link_performance_sampling_interval_in_min",
        "td_link_performance_sampling_interval_hd_in_min",
    ])
    rows.append(["", "", "1", "0.0001", "1", "1", "0", "", "60", "15"])
    rows.append([])

    # [metric]
    rows.append(["[metric]", "", "speed_mph_flag"])
    rows.append(["", "", "1"])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)


def _align_zones_to_network(demand_dir, input_dir):
    """
    Reconcile grid2demand zone IDs with the road network.

    grid2demand creates its own grid-based zone IDs (0, 1, 2...) that don't
    match what osm2gmns put in node.csv.  This reads grid2demand's zone.csv
    (centroid locations) and demand.csv (which zone IDs are actually used),
    then finds the nearest network node for each demand zone and writes an
    updated node.csv where those nodes carry the grid2demand zone_ids.
    DTALite can then route demand correctly.
    """
    g2d_zone_csv = demand_dir / "zone.csv"
    demand_csv = demand_dir / "demand.csv"
    src_node_csv = input_dir / "node.csv"

    if not g2d_zone_csv.exists() or not src_node_csv.exists():
        print("[warn] zone.csv or node.csv missing — skipping zone alignment, copying node.csv as-is")
        shutil.copy2(src_node_csv, demand_dir / "node.csv")
        return

    # Load grid2demand zone centroids
    zone_centroids = {}
    with open(g2d_zone_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                zid = int(row["id"].strip())
                cx = float(row["centroid_x"].strip())
                cy = float(row["centroid_y"].strip())
                zone_centroids[zid] = (cx, cy)
            except (ValueError, KeyError):
                continue

    # Find which zone IDs are actually referenced in demand.csv
    demand_zone_ids = set()
    if demand_csv.exists():
        with open(demand_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    demand_zone_ids.add(int(row["o_zone_id"]))
                    demand_zone_ids.add(int(row["d_zone_id"]))
                except (ValueError, KeyError):
                    pass

    zones_to_map = {z: zone_centroids[z] for z in demand_zone_ids if z in zone_centroids}

    # Read all network nodes
    with open(src_node_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        nodes = [dict(r) for r in reader]

    # For each demand zone, find nearest network node
    node_zone = {}  # node_id -> zone_id
    for zone_id, (cx, cy) in zones_to_map.items():
        best_id, best_dist = None, float("inf")
        for node in nodes:
            try:
                nx = float(node.get("x_coord") or 0)
                ny = float(node.get("y_coord") or 0)
                d = (nx - cx) ** 2 + (ny - cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_id = node.get("node_id", "")
            except (ValueError, TypeError):
                continue
        if best_id is not None:
            node_zone[best_id] = zone_id

    # Write updated node.csv: clear old zone_ids, assign grid2demand ones
    out_node = demand_dir / "node.csv"
    with open(out_node, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for node in nodes:
            nid = node.get("node_id", "")
            node["zone_id"] = str(node_zone[nid]) if nid in node_zone else ""
            writer.writerow(node)

    print(f"[ok] node.csv written — {len(node_zone)} zone centroids aligned to network nodes")


def _write_zone_csv(node_csv, out_path):
    """Write zone.csv in DLSim format from node.csv zone_id column."""
    zones = {}
    if node_csv.exists():
        with open(node_csv, newline="") as f:
            for row in csv.DictReader(f):
                raw = row.get("zone_id", "").strip()
                if raw and raw.isdigit():
                    zid = int(raw)
                    if zid > 0:
                        nid = row.get("node_id", "").strip()
                        if nid:
                            zones.setdefault(zid, nid)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["zone_id", "x_coord", "y_coord", "access_node_vector"])
        for zone_id, node_id in sorted(zones.items()):
            writer.writerow([zone_id, "", "", node_id])
    print(f"[ok] zone.csv written ({len(zones)} zones)")


def run(input_folder, demand_period="AM", time_period="0700_0800"):
    input_dir = pathlib.Path(input_folder).resolve()
    if not input_dir.is_dir():
        print(f"[error] Directory not found: {input_dir}")
        sys.exit(1)

    link_csv = input_dir / "link.csv"
    if not link_csv.exists():
        print(f"[error] link.csv not found in {input_dir}")
        sys.exit(1)

    demand_dir = input_dir.parent / f"{input_dir.name}_demand"
    if not demand_dir.exists():
        print(f"[warn] Demand folder not found: {demand_dir}")
        print("   Creating it now (run run_grid2demand.py first to populate demand.csv)")
        demand_dir.mkdir(parents=True)

    # Copy link.csv
    shutil.copy2(link_csv, demand_dir / "link.csv")
    print(f"Copied link.csv → {demand_dir.name}/")

    # Align grid2demand zones to network nodes, write node.csv
    _align_zones_to_network(demand_dir, input_dir)

    # Write DLSim zone.csv from the updated node.csv
    _write_zone_csv(demand_dir / "node.csv", demand_dir / "zone.csv")

    link_types = read_link_types(link_csv)
    print(
        f"Found {len(link_types)} link type(s): "
        + ", ".join(f"{lt} ({name})" for lt, name, _ in link_types)
    )

    settings_path = demand_dir / "settings.csv"
    write_settings_csv(settings_path, link_types, demand_period, time_period)
    print(f"[ok] settings.csv written to: {settings_path}")
    print(f"\nDLSim-MRM input package: {demand_dir}/")
    print("   node.csv, link.csv, demand.csv, zone.csv, settings.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate settings.csv for DLSim-MRM")
    parser.add_argument("folder", nargs="?", default=None,
                        help="Input folder containing node.csv and link.csv (e.g. data/14850)")
    parser.add_argument("--period", default="AM", help="Demand period label (default: AM)")
    parser.add_argument("--time", default="0700_0800",
                        help="Time window in HHMM_HHMM format (default: 0700_0800)")
    args = parser.parse_args()

    folder = args.folder
    if not folder:
        folder = input("Enter input folder name (e.g. 'data/14850'): ").strip()
        if not folder:
            print("No folder provided. Exiting.")
            sys.exit(0)

    run(folder, demand_period=args.period, time_period=args.time)
