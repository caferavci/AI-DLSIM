#!/usr/bin/env python3
"""Fetch OSM-based roads+POI and generate CSVs via osm2gmns."""

import os
import sys
import shutil
import json
import time
import pathlib
import requests
from typing import Optional
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL     = "gpt-4o"

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
NOMINATIM_URL    = "https://nominatim.openstreetmap.org/search"

MAX_QUERY_RETRIES    = 3   # How many times to ask the LLM to fix a bad query
OVERPASS_TIMEOUT_SEC = 120 # Overpass server-side timeout (injected into query)

ROAD_HIGHWAY_TAGS = [
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link", "living_street",
    "service", "road"
]

# ── Overpass query documentation injected into every LLM prompt ──────────────

OVERPASS_DOCS = """
You are an expert in OpenStreetMap Overpass QL.

OVERPASS QL RULES YOU MUST FOLLOW:
1. Every query starts with a settings block:  [out:xml][timeout:120];
2. Use a union block  ( ... );  to group multiple statement types.
3. After the union, always end with:  out body; >; out skel qt;
4. Bounding box syntax is:  (south,west,north,east)  — all decimal degrees, NO quotes.
5. To filter by tag:  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|service|road"](bbox);
6. 'bbox' is a literal keyword when you have defined  [bbox:S,W,N,E]  in the settings.
   OR you can inline the bbox directly:  way[...](south,west,north,east);
7. node and relation statements must mirror the way statement when using >; to resolve geometry.
8. DO NOT use Overpass Turbo macros like {{bbox}} — raw coordinates only.
MINIMAL CORRECT TEMPLATE (bbox):
[out:xml][timeout:120];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|service|road"](S,W,N,E);
  node(w);
  relation["highway"](S,W,N,E);
  node["amenity"](S,W,N,E);
  node["shop"](S,W,N,E);
  node["building"](S,W,N,E);
  node["leisure"](S,W,N,E);
  node["tourism"](S,W,N,E);
  way["amenity"](S,W,N,E);
  way["shop"](S,W,N,E);
  way["building"](S,W,N,E);
  way["leisure"](S,W,N,E);
  way["tourism"](S,W,N,E);
);
out body;
>;
out skel qt;

Replace S,W,N,E with actual decimal-degree numbers (no quotes, no spaces inside parentheses beyond the commas).

POLYGON FILTER (use instead of bbox when an exact boundary polygon is available):
[out:xml][timeout:120];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|service|road"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  node(w);
  relation["highway"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  node["amenity"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  node["shop"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  node["building"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  node["leisure"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  node["tourism"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  way["amenity"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  way["shop"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  way["building"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  way["leisure"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
  way["tourism"](poly:"lat1 lon1 lat2 lon2 lat3 lon3 ...");
);
out body;
>;
out skel qt;

The poly value is a space-separated string of alternating lat lon pairs (no commas between pairs).
"""

# ── Helper: resolve location to bounding box or polygon ──────────────────────


def is_zipcode(text: str) -> bool:
    return text.strip().isdigit() and len(text.strip()) == 5



def bbox_from_nominatim(location: str) -> Optional[dict]:
    """Ask Nominatim (OSM) to geocode a free-text location."""
    params = {
        "q": location,
        "format": "json",
        "limit": 1,
    }
    headers = {"User-Agent": "OSMRoadFetcher/1.0 (traffic-simulation-research)"}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        results = r.json()
        if not results:
            return None
        best = results[0]
        bb = best.get("boundingbox")   # [S, N, W, E]  ← Nominatim order
        if bb:
            return {
                "south": float(bb[0]),
                "north": float(bb[1]),
                "west":  float(bb[2]),
                "east":  float(bb[3]),
                "display_name": best.get("display_name", location),
            }
    except Exception as e:
        print(f"  [Nominatim error] {e}")
    return None


def resolve_location(user_input: str) -> dict:
    """
    Return a location dict for whatever the user typed.
    For ZIP codes, attempts to fetch a precise ZCTA polygon from Census TIGER.
    Falls back to Nominatim bbox for everything else (or if TIGER fails).

    Returned dict always has 'display_name'. It will have either:
      - 'poly': str   — space-separated "lat lon ..." pairs for Overpass poly:
      - 'south', 'north', 'west', 'east': float — bounding box
    """
    text = user_input.strip()
    print(f"\nResolving location: '{text}'")

    bbox = bbox_from_nominatim(text + ", USA" if is_zipcode(text) else text)
    if not bbox:
        print("[error] Could not resolve location to coordinates. Please try a different query.")
        sys.exit(1)

    print(f"   → {bbox['display_name']}")
    print(f"   → Bounding box: S={bbox['south']:.4f}  N={bbox['north']:.4f}  "
          f"W={bbox['west']:.4f}  E={bbox['east']:.4f}")
    return bbox


# ── LLM query generation ──────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return (
        OVERPASS_DOCS
        + "\n\nYou will be given either a bounding box (south, west, north, east) "
        "OR a polygon string for use with the Overpass poly: filter. "
        "Produce ONLY a valid Overpass QL query string — no explanation, "
        "no markdown fences, no comments. Just the raw query text."
    )


def ask_llm_for_query(client: OpenAI, location: dict,
                      previous_error: Optional[str] = None) -> str:
    """Call OpenAI to generate (or fix) an Overpass QL query."""
    if "poly" in location:
        area_desc = (
            f"Use the following polygon string for the Overpass poly: filter:\n"
            f"  {location['poly']}\n\n"
            "Use poly: instead of a bounding box in the query."
        )
    else:
        s = location["south"]
        w = location["west"]
        n = location["north"]
        e = location["east"]
        area_desc = f"Use the bounding box: south={s}, west={w}, north={n}, east={e}"

    user_msg = (
        f"Generate an Overpass QL query to fetch ALL roads AND points of interest (POIs) in this area.\n"
        f"{area_desc}\n\n"
        "Include:\n"
        "1. All highway types relevant for traffic simulation: "
        "motorway, trunk, primary, secondary, tertiary, unclassified, "
        "residential, *_link, living_street, service.\n"
        "2. POI nodes and ways tagged with: amenity, shop, building, leisure, tourism.\n"
        "Return the raw query only."
    )

    if previous_error:
        user_msg += (
            f"\n\nThe previous query failed with this error:\n{previous_error}\n"
            "Fix the query so it is syntactically correct Overpass QL."
        )

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": user_msg},
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()

    return raw


# ── Overpass query execution ──────────────────────────────────────────────────

def run_overpass_query(query: str) -> Optional[requests.Response]:
    """
    POST the query to the Overpass API via curl (bypasses Python LibreSSL issues),
    trying each endpoint until one succeeds.
    Returns a minimal Response-like object with .status_code, .content, and .text.
    """
    import subprocess

    class _CurlResponse:
        def __init__(self, status_code: int, content: bytes):
            self.status_code = status_code
            self.content = content
            self.text = content.decode("utf-8", errors="replace")

    for url in OVERPASS_URLS:
        print(f"\nSending query to {url} …")
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-w", "\n__HTTP_STATUS__%{http_code}",
                    "-X", "POST", url,
                    "--data-urlencode", f"data={query}",
                    "--max-time", str(OVERPASS_TIMEOUT_SEC + 30),
                ],
                capture_output=True,
                timeout=OVERPASS_TIMEOUT_SEC + 60,
            )
            raw = result.stdout
            # Split body and status code appended by -w
            marker = b"\n__HTTP_STATUS__"
            if marker in raw:
                body, status_str = raw.rsplit(marker, 1)
                status_code = int(status_str.strip())
            else:
                body = raw
                status_code = 0

            if status_code == 200:
                return _CurlResponse(200, body)
            print(f"  [warn] {url} returned HTTP {status_code} — trying next endpoint…")
        except subprocess.TimeoutExpired:
            print(f"  [warn] {url} timed out — trying next endpoint…")
        except Exception as e:
            print(f"  [warn] {url} failed: {e} — trying next endpoint…")

    print("  [error] All Overpass endpoints failed.")
    return None


def overpass_error_message(response: requests.Response) -> Optional[str]:
    """Extract a human-readable error from an Overpass response, or None if OK."""
    if response.status_code == 200:
        text = response.text
        if "<remark>" in text:
            # Overpass embeds runtime errors as <remark> elements
            start = text.index("<remark>") + len("<remark>")
            end   = text.index("</remark>")
            return text[start:end].strip()
        if text.strip().startswith("<?xml") or text.strip().startswith("<osm"):
            return None   # looks good
        return f"Unexpected response body (first 300 chars): {text[:300]}"
    return f"HTTP {response.status_code}: {response.text[:300]}"


# ── osm2gmns conversion ───────────────────────────────────────────────────────

def build_multiresolution_nets(osm_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Convert an .osm file to macro/meso/micro networks via osm2gmns."""
    print("\nConverting OSM data to macro / meso / micro networks via osm2gmns…")
    try:
        import osm2gmns as og

        net = og.getNetFromFile(
            str(osm_path),
            network_types=("auto",),
            POI=True,
            default_lanes=True,
            default_speed=True,
            default_capacity=True,
        )
        og.consolidateComplexIntersections(net, auto_identify=True)
        og.generateNodeActivityInfo(net)
        og.buildMultiResolutionNets(net)
        og.outputNetToCSV(net, output_folder=str(out_dir))

        # Copy root-level files into each net subfolder
        for net_dir in (out_dir / "macronet", out_dir / "mesonet", out_dir / "micronet"):
            if net_dir.exists():
                for fname in ("node.csv", "link.csv", "movement.csv", "poi.csv"):
                    src = out_dir / fname
                    if src.exists() and not (net_dir / fname).exists():
                        shutil.copy2(src, net_dir / fname)

        print(f"   [ok] macronet → {out_dir}/macronet/")
        print(f"   [ok] mesonet  → {out_dir}/mesonet/")
        print(f"   [ok] micronet → {out_dir}/micronet/")

    except ImportError:
        print("[warn] osm2gmns is not installed. Run:  pip install osm2gmns==0.7.5")
        print(f"   The .osm file is saved at {osm_path}")
    except Exception as e:
        print(f"[warn] osm2gmns conversion failed: {e}")
        print(f"   The .osm file is still saved at {osm_path}")


# ── Main flow ─────────────────────────────────────────────────────────────────

def main():
    # ── API key check
    if not OPENAI_API_KEY:
        print("Set the OPENAI_API_KEY environment variable first.")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    # ── User input
    print("=" * 60)
    print("  OSM Road Fetcher — Traffic Simulation Data Tool")
    print("=" * 60)
    user_input = input("\nEnter a location or ZIP code: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
        sys.exit(0)

    # ── Resolve to bounding box or polygon
    location = resolve_location(user_input)

    # ── LLM query generation + retry loop
    query       = None
    last_error  = None

    for attempt in range(1, MAX_QUERY_RETRIES + 1):
        print(f"\nAsking {OPENAI_MODEL} to {'generate' if attempt == 1 else 'fix'} "
              f"the Overpass query (attempt {attempt}/{MAX_QUERY_RETRIES})…")

        query = ask_llm_for_query(client, location, previous_error=last_error)

        print("\n── Generated query ──────────────────────────────────────")
        print(query)
        print("─────────────────────────────────────────────────────────")

        response = run_overpass_query(query)
        if response is None:
            last_error = "Network error — could not reach Overpass API."
            continue

        error = overpass_error_message(response)
        if error is None:
            print("[ok] Query succeeded!")
            break
        else:
            print(f"[warn] Query error: {error}")
            last_error = error
            if attempt < MAX_QUERY_RETRIES:
                print("   Asking the LLM to fix the query…")
                time.sleep(1)
    else:
        print("\nAll attempts failed. Last error:", last_error)
        print("   Try running the script again or providing a more specific location.")
        sys.exit(1)

    # Determine output location in repository data folder
    safe_name = (
        user_input.strip()
                  .lower()
                  .replace(" ", "_")
                  .replace(",", "")
                  .replace("/", "-")
    )[:50]
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    out_dir = repo_root / "data" / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput folder: {out_dir}/")

    # ── Save .osm file
    output_path = out_dir / f"{safe_name}_roads.osm"
    with open(output_path, "wb") as f:
        f.write(response.content)

    size_mb = len(response.content) / 1_048_576
    print(f"Saved {size_mb:.2f} MB → {output_path}")

    # ── Convert .osm → macro / meso / micro networks via osm2gmns
    build_multiresolution_nets(output_path, out_dir)


if __name__ == "__main__":
    main()