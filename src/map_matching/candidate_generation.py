from pathlib import Path

import osmnx as ox
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "road_network"
    / "beijing_drive.graphml"
)


# ============================================================
# Candidate generation configuration
# ============================================================

SEARCH_RADIUS_METERS = 100
NUMBER_OF_CANDIDATES = 5


# ============================================================
# Load and prepare road network
# ============================================================

def load_road_network():

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Road network not found:\n{GRAPH_FILE}"
        )

    print("Loading road network...")

    G = ox.load_graphml(GRAPH_FILE)

    print(
        f"Loaded graph: "
        f"{len(G.nodes):,} nodes, "
        f"{len(G.edges):,} edges"
    )

    # Project graph from EPSG:4326 to a local metric CRS.
    # This allows us to calculate distances in meters.
    G_projected = ox.project_graph(G)

    print(
        f"Projected CRS: "
        f"{G_projected.graph.get('crs')}"
    )

    # Convert road edges to GeoDataFrame.
    edges = ox.graph_to_gdfs(
        G_projected,
        nodes=False,
        edges=True
    )

    return G_projected, edges


# ============================================================
# Coordinate transformer
# ============================================================

def create_coordinate_transformer(projected_crs):

    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True
    )

    return transformer


# ============================================================
# Find candidate road segments
# ============================================================

def find_candidates(
    latitude,
    longitude,
    edges,
    transformer,
    radius_meters=SEARCH_RADIUS_METERS,
    max_candidates=NUMBER_OF_CANDIDATES
):
    """
    Find the closest road segments to one GPS observation.

    Parameters
    ----------
    latitude : float
        GPS latitude.

    longitude : float
        GPS longitude.

    edges : GeoDataFrame
        Projected OSM road edges.

    transformer : Transformer
        EPSG:4326 -> projected CRS transformer.

    radius_meters : float
        Maximum search radius around GPS point.

    max_candidates : int
        Maximum number of road candidates to return.

    Returns
    -------
    list of dictionaries
        Candidate road information.
    """

    # Convert GPS coordinates into projected coordinates.
    x, y = transformer.transform(
        longitude,
        latitude
    )

    gps_point = Point(x, y)

    # Create a search area around the GPS point.
    search_area = gps_point.buffer(
        radius_meters
    )

    # Use the spatial index to find nearby edges.
    possible_indices = edges.sindex.query(
        search_area,
        predicate="intersects"
    )

    if len(possible_indices) == 0:
        return []

    candidates = []

    for index in possible_indices:

        edge = edges.iloc[index]

        geometry = edge.geometry

        distance = gps_point.distance(
            geometry
        )

        if distance <= radius_meters:

            u, v, key = edge.name

            candidates.append(
                {
                    "u": u,
                    "v": v,
                    "key": key,
                    "distance_meters": float(distance),
                    "road_type": edge.get(
                        "highway",
                        None
                    ),
                    "road_name": edge.get(
                        "name",
                        None
                    ),
                    "maxspeed": edge.get(
                        "maxspeed",
                        None
                    ),
                    "length_meters": edge.get(
                        "length",
                        None
                    ),
                }
            )

    # Sort by distance from GPS point.
    candidates.sort(
        key=lambda candidate:
        candidate["distance_meters"]
    )

    # Keep only the closest candidates.
    return candidates[:max_candidates]


# ============================================================
# Find a test GPS point
# ============================================================

def find_test_gps_point():

    processed_root = (
        PROJECT_ROOT / "data" / "processed"
    )

    # Search processed trajectories.
    for csv_file in processed_root.glob(
        "user_*/*.csv"
    ):

        # Skip analysis CSVs.
        if "region_analysis" in csv_file.parts:
            continue

        try:

            df = pd.read_csv(
                csv_file,
                nrows=100
            )

        except Exception:
            continue

        if df.empty:
            continue

        # Find a point inside our Beijing test area.
        selected = df[
            (df["latitude"] >= 39.84)
            & (df["latitude"] <= 40.04)
            & (df["longitude"] >= 116.29)
            & (df["longitude"] <= 116.49)
        ]

        if not selected.empty:

            row = selected.iloc[0]

            return {
                "user_id": row["user_id"],
                "trajectory_id": row[
                    "trajectory_id"
                ],
                "timestamp": row["timestamp"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
            }

    return None


# ============================================================
# Main test
# ============================================================

def main():

    print("=" * 60)
    print("GPS → ROAD CANDIDATE GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load road network
    # --------------------------------------------------------

    G, edges = load_road_network()

    # --------------------------------------------------------
    # Create coordinate transformer
    # --------------------------------------------------------

    transformer = create_coordinate_transformer(
        G.graph["crs"]
    )

    # --------------------------------------------------------
    # Find a real GeoLife GPS point
    # --------------------------------------------------------

    gps_point = find_test_gps_point()

    if gps_point is None:

        raise RuntimeError(
            "Could not find a GeoLife GPS point "
            "inside the Beijing test area."
        )

    print("\nTest GPS observation:")

    print(
        f"  User       : "
        f"{gps_point['user_id']}"
    )

    print(
        f"  Trajectory : "
        f"{gps_point['trajectory_id']}"
    )

    print(
        f"  Timestamp  : "
        f"{gps_point['timestamp']}"
    )

    print(
        f"  Latitude   : "
        f"{gps_point['latitude']}"
    )

    print(
        f"  Longitude  : "
        f"{gps_point['longitude']}"
    )

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidates = find_candidates(
        latitude=gps_point["latitude"],
        longitude=gps_point["longitude"],
        edges=edges,
        transformer=transformer
    )

    print(
        f"\nCandidates found: "
        f"{len(candidates)}"
    )

    if not candidates:

        print(
            "No road candidates found "
            "within the search radius."
        )

        return

    print("\nCandidate roads:")

    for i, candidate in enumerate(
        candidates,
        start=1
    ):

        print(
            f"\n  Candidate {i}"
        )

        print(
            f"    Edge       : "
            f"({candidate['u']}, "
            f"{candidate['v']}, "
            f"{candidate['key']})"
        )

        print(
            f"    Distance   : "
            f"{candidate['distance_meters']:.2f} m"
        )

        print(
            f"    Road type  : "
            f"{candidate['road_type']}"
        )

        print(
            f"    Road name  : "
            f"{candidate['road_name']}"
        )

        print(
            f"    Road length: "
            f"{candidate['length_meters']}"
        )

    print("\n" + "=" * 60)
    print("CANDIDATE GENERATION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
