from pathlib import Path

import osmnx as ox


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "road_network"
OUTPUT_FILE = OUTPUT_DIR / "beijing_drive.graphml"


# ============================================================
# Focused Beijing test area
# ============================================================

# OSMnx expects:
# (left, bottom, right, top)
# = (min_longitude, min_latitude, max_longitude, max_latitude)

BBOX = (
    116.29,   # left / minimum longitude
    39.84,    # bottom / minimum latitude
    116.49,   # right / maximum longitude
    40.04,    # top / maximum latitude
)


# ============================================================
# Download road network
# ============================================================

def download_road_network():

    print("=" * 60)
    print("BEIJING OSM ROAD NETWORK DOWNLOAD")
    print("=" * 60)

    print("\nRequested bounding box:")
    print(f"  Left      : {BBOX[0]}")
    print(f"  Bottom    : {BBOX[1]}")
    print(f"  Right     : {BBOX[2]}")
    print(f"  Top       : {BBOX[3]}")

    print("\nNetwork type: drive")
    print("Downloading road network from OpenStreetMap...")
    print("This may take some time.\n")

    # Download drivable road network
    G = ox.graph.graph_from_bbox(
        BBOX,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    print("\nDownload completed.")

    print(f"Nodes: {len(G.nodes):,}")
    print(f"Edges: {len(G.edges):,}")

    # Create output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save graph
    ox.io.save_graphml(
        G,
        filepath=OUTPUT_FILE
    )

    print("\nRoad network saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("OSM ROAD NETWORK SETUP COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    download_road_network()