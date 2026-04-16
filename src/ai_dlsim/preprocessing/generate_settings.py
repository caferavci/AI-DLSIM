"""
generate_settings.py

Generates settings.csv for DLSim-MRM from a preprocessed data folder.

Reads link.csv to discover which link types are present, then writes settings.csv
into the sibling <folder>_demand directory (where demand.csv from run_grid2demand.py
lives).  Also copies node.csv and link.csv there so the _demand folder is a
complete, self-contained DLSim-MRM input package.

Usage:
    python generate_settings.py
    python generate_settings.py data/14850
    python generate_settings.py data/14850 --period PM --time 1500_1800
"""

import sys
import csv
import shutil
import pathlib

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


def read_link_types(link_csv: pathlib.Path) -> list[tuple[int, str, str]]:
    """Return sorted list of (link_type_id, name, type_code) found in link.csv."""
    seen: dict[int, str] = {}
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


def write_settings_csv(
    out_path: pathlib.Path,
    link_types: list[tuple[int, str, str]],
    demand_period: str,
    time_period: str,
) -> None:
    rows: list[list[str]] = []

    # [assignment]
    rows.append([
        "[assignment]", "assignment_mode", "column_generation_iterations",
        "column_updating_iterations", "odme_iterations",
        "simulation_iterations", "number_of_memory_blocks",
    ])
    rows.append(["", "dta", "20", "20", "400", "0", "4"])
    rows.append([])

    # [agent_type]
    rows.append([
        "[agent_type]", "agent_type", "name", "", "vot",
        "flow_type", "pce", "occ",
    ])
    rows.append(["", "sov",  "passenger",          "", "10", "0", "1",   "1"])
    rows.append(["", "hov2", "HOV2",               "", "10", "0", "2",   "2"])
    rows.append(["", "hov3", "HOV3",               "", "10", "0", "1",   "3.5"])
    rows.append(["", "sut",  "Single-Unit Trucks", "", "10", "0", "1",   "1"])
    rows.append(["", "mut",  "Multi-Unit Trucks",  "", "10", "0", "4",   "1"])
    rows.append([])

    # [link_type]
    rows.append([
        "[link_type]", "link_type", "link_type_name", "",
        "agent_type_blocklist", "type_code", "traffic_flow_code", "vdf_type",
    ])
    for lt, name, type_code in link_types:
        rows.append(["", str(lt), name, "", "", type_code, "0", "qvdf"])
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
    rows.append(["", "1", "demand.csv", "", "column", demand_period, "sov", "1"])
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
        writer = csv.writer(f)
        writer.writerows(rows)


def run(input_folder: str, demand_period: str = "AM", time_period: str = "0700_0800") -> None:
    input_dir = pathlib.Path(input_folder).resolve()
    if not input_dir.is_dir():
        print(f"[error] Directory not found: {input_dir}")
        sys.exit(1)

    link_csv = input_dir / "link.csv"
    if not link_csv.exists():
        print(f"[error] link.csv not found in {input_dir}")
        sys.exit(1)

    # Output goes to the sibling _demand folder produced by run_grid2demand.py
    demand_dir = input_dir.parent / f"{input_dir.name}_demand"
    if not demand_dir.exists():
        print(f"[warn] Demand folder not found: {demand_dir}")
        print("   Creating it now (run run_grid2demand.py first to populate demand.csv)")
        demand_dir.mkdir(parents=True)

    # Copy node.csv and link.csv so the _demand folder is a complete DLSim input package
    for fname in ("node.csv", "link.csv"):
        src = input_dir / fname
        if src.exists():
            shutil.copy2(src, demand_dir / fname)
            print(f"Copied {fname} → {demand_dir.name}/")
        else:
            print(f"[warn] {fname} not found in {input_dir} — skipping copy")

    link_types = read_link_types(link_csv)
    print(
        f"Found {len(link_types)} link type(s): "
        + ", ".join(f"{lt} ({name})" for lt, name, _ in link_types)
    )

    settings_path = demand_dir / "settings.csv"
    write_settings_csv(settings_path, link_types, demand_period, time_period)
    print(f"[ok] settings.csv written to: {settings_path}")
    print(f"\nDLSim-MRM input package: {demand_dir}/")
    print("   node.csv, link.csv, demand.csv, settings.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate settings.csv for DLSim-MRM")
    parser.add_argument(
        "folder", nargs="?", default=None,
        help="Input folder containing node.csv and link.csv (e.g. data/14850)"
    )
    parser.add_argument(
        "--period", default="AM",
        help="Demand period label (default: AM)"
    )
    parser.add_argument(
        "--time", default="0700_0800",
        help="Time window in HHMM_HHMM format (default: 0700_0800)"
    )
    args = parser.parse_args()

    folder = args.folder
    if not folder:
        folder = input("Enter input folder name (e.g. 'data/14850'): ").strip()
        if not folder:
            print("No folder provided. Exiting.")
            sys.exit(0)

    run(folder, demand_period=args.period, time_period=args.time)
