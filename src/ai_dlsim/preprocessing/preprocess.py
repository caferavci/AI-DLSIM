"""
preprocess.py

Full preprocessing pipeline: natural language query → DLSim-MRM input package.

    Parse    llm_query_parser  — natural language → structured QueryRequest
    Step 1   retrieve_csvs     — fetch OSM data, convert to node/link/poi CSVs
    Step 2   run_grid2demand   — generate demand.csv via gravity model
    Step 3   generate_settings — write settings.csv, assemble final package

Usage:
    python preprocess.py                          # interactive, uses macronet
    python preprocess.py "zip code 14850"         # non-interactive, macronet
    python preprocess.py "zip code 14850" --net mesonet
    python preprocess.py "zip code 14850" --net micronet
"""

import sys
import argparse
import pathlib
from typing import Optional, Tuple

VALID_NETS = ("macronet", "mesonet", "micronet")

_here = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_here.parent))           # preprocessing siblings
sys.path.insert(0, str(_here.parents[2]))       # src/ → ai_dlsim package

import retrieve_csvs as _rc
import run_grid2demand as _g2d
import generate_settings as _gs
from llm_query_parser import LlmQueryParser


def _safe_name(user_input: str) -> str:
    """Replicate retrieve_csvs folder-naming logic."""
    return (
        user_input.strip()
                  .lower()
                  .replace(" ", "_")
                  .replace(",", "")
                  .replace("/", "-")
    )[:50]


def _departure_time_to_period(departure_time: Optional[str]) -> Tuple[str, str]:
    """
    Map a HH:MM departure time to a (period_label, time_window) pair.
    Returns AM / 0700_0800 if departure_time is None or unrecognised.
    """
    if not departure_time:
        return "AM", "0700_0800"
    try:
        hour = int(departure_time.split(":")[0])
    except (ValueError, IndexError):
        return "AM", "0700_0800"

    if 6 <= hour < 9:
        return "AM", "0600_0900"
    if 9 <= hour < 15:
        return "MD", "0900_1500"
    if 15 <= hour < 19:
        return "PM", "1500_1800"
    return "AM", "0700_0800"


def main() -> None:
    parser = argparse.ArgumentParser(description="DLSim-MRM Preprocessing Pipeline")
    parser.add_argument("query", nargs="?", default=None,
                        help="Simulation scenario query or ZIP code (interactive if omitted)")
    parser.add_argument("--net", choices=VALID_NETS, default="macronet",
                        help="Network resolution to use for demand generation (default: macronet)")
    args = parser.parse_args()
    net = args.net

    print("=" * 60)
    print("  DLSim-MRM Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Network: {net}")

    if args.query:
        user_query = args.query
    else:
        user_query = input("\nDescribe your simulation scenario: ").strip()
    if not user_query:
        print("No input provided. Exiting.")
        sys.exit(0)

    # If the input is (or contains) a bare ZIP code, use it directly so the
    # ZCTA polygon path in resolve_location fires correctly.
    zip_match = next((w for w in user_query.split() if w.isdigit() and len(w) == 5), None)
    if zip_match:
        print(f"\nZIP code detected ({zip_match}), skipping LLM parse.")
        region = zip_match
        departure_time = None
    else:
        print("\nParsing query…")
        request = LlmQueryParser().parse(user_query)
        print(f"   region          : {request.region}")
        print(f"   departure_time  : {request.departure_time}")
        print(f"   mode            : {request.mode}")
        print(f"   scenario        : {request.scenario}")
        region = request.region
        departure_time = request.departure_time

    demand_period, time_period = _departure_time_to_period(departure_time)
    print(f"   → demand period : {demand_period} ({time_period})")

    repo_root = _here.parents[3]
    data_dir = repo_root / "data" / _safe_name(region)

    print("\n" + "─" * 60)
    print("  Step 1/3 — Fetching OSM data and generating CSVs")
    print("─" * 60)

    if not _rc.OPENAI_API_KEY:
        print("Set the OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=_rc.OPENAI_API_KEY)

    location = _rc.resolve_location(region)

    last_error = None
    response = None
    for attempt in range(1, _rc.MAX_QUERY_RETRIES + 1):
        print(f"\nAsking {_rc.OPENAI_MODEL} to {'generate' if attempt == 1 else 'fix'} "
              f"the Overpass query (attempt {attempt}/{_rc.MAX_QUERY_RETRIES})…")
        query = _rc.ask_llm_for_query(client, location, previous_error=last_error)
        print("\n── Generated query ──────────────────────────────────────")
        print(query)
        print("─────────────────────────────────────────────────────────")
        response = _rc.run_overpass_query(query)
        if response is None:
            last_error = "Network error — could not reach Overpass API."
            continue
        error = _rc.overpass_error_message(response)
        if error is None:
            print("[ok] Query succeeded!")
            break
        print(f"[warn] Query error: {error}")
        last_error = error
    else:
        print("\nAll Overpass attempts failed. Last error:", last_error)
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)
    osm_path = data_dir / f"{_safe_name(region)}_roads.osm"
    with open(osm_path, "wb") as f:
        f.write(response.content)
    print(f"Saved {len(response.content) / 1_048_576:.2f} MB → {osm_path}")

    _rc.build_multiresolution_nets(osm_path, data_dir)

    print("\n" + "─" * 60)
    print(f"  Step 2/3 — Generating demand via grid2demand ({net})")
    print("─" * 60)
    _g2d.run(str(data_dir / net))

    print("\n" + "─" * 60)
    print("  Step 3/3 — Generating settings.csv")
    print("─" * 60)
    _gs.run(str(data_dir / net), demand_period=demand_period, time_period=time_period)

    demand_dir = data_dir / f"{net}_demand"
    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  DLSim-MRM input package: {demand_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
