"""
run_grid2demand.py

Given a folder name (e.g. '14850') that contains node.csv, link.csv, and poi.csv
(produced by retrieve_csvs.py), runs grid2demand and writes all outputs to a new
sibling folder named '<folder>_demand'.

Usage:
    python39 run_grid2demand.py
    python39 run_grid2demand.py 14850
"""

import sys
import shutil
import pathlib
import grid2demand as gd
import grid2demand.func_lib.gen_zone as _gen_zone


# Monkey patch grid2demand bug in get_lng_lat_min_max for non-zero node IDs.
def _fixed_get_lng_lat_min_max(node_dict):
    first = next(iter(node_dict.values()))
    coord_x_min = coord_x_max = first.x_coord
    coord_y_min = coord_y_max = first.y_coord
    for node in node_dict.values():
        coord_x_min = min(coord_x_min, node.x_coord)
        coord_x_max = max(coord_x_max, node.x_coord)
        coord_y_min = min(coord_y_min, node.y_coord)
        coord_y_max = max(coord_y_max, node.y_coord)
    return [coord_x_min - 1e-6, coord_x_max + 1e-6, coord_y_min - 1e-6, coord_y_max + 1e-6]

_gen_zone.get_lng_lat_min_max = _fixed_get_lng_lat_min_max


def run(input_folder: str) -> None:
    input_dir = pathlib.Path(input_folder).resolve()
    if not input_dir.is_dir():
        print(f"❌ Directory not found: {input_dir}")
        sys.exit(1)

    output_dir = input_dir.parent / f"{input_dir.name}_demand"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    print(f"📂 Input : {input_dir}")
    print(f"📂 Output: {output_dir}\n")

    net = gd.GRID2DEMAND(input_dir=str(input_dir), output_dir=str(output_dir))
    network = net.load_network
    node_dict = network["node_dict"]
    poi_dict = network["poi_dict"]

    zone_dict = net.net2zone(node_dict)
    net.sync_geometry_between_zone_and_node_poi(zone_dict, node_dict, poi_dict)
    net.calc_zone_od_distance_matrix(zone_dict)
    net.gen_poi_trip_rate(poi_dict)
    net.gen_node_prod_attr(node_dict, poi_dict)
    net.calc_zone_prod_attr(node_dict, zone_dict)
    net.run_gravity_model()

    net.save_demand
    net.save_zone
    net.save_zone_od_dist_table

    print(f"\n✅ Done. Outputs written to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = input("Enter input folder name (e.g. '14850'): ").strip()
        if not folder:
            print("No folder provided. Exiting.")
            sys.exit(0)

    run(folder)
