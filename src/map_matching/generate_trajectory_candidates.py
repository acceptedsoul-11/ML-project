from pathlib import Path

import pandas as pd
import osmnx as ox

from candidate_generation import (
    create_coordinate_transformer,
    find_candidates,
)


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

TRAJECTORY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "user_000"
    / "20081023025304.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "map_matching"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "20081023025304_candidates.csv"
)


# ============================================================
# Configuration
# ============================================================

SEARCH_RADIUS_METERS = 100
MAX_CANDIDATES = 5


# ============================================================
# Load road network
# ============================================================

def load_road_network():

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Road network not found:\n{GRAPH_FILE}"
        )

    print("Loading OSM road network...")

    G = ox.load_graphml(
        GRAPH_FILE
    )

    print(
        f"Loaded graph: "
        f"{len(G.nodes):,} nodes, "
        f"{len(G.edges):,} edges"
    )

    # Project graph into a metric CRS.
    G_projected = ox.project_graph(G)

    # Convert edges into GeoDataFrame.
    edges = ox.graph_to_gdfs(
        G_projected,
        nodes=False,
        edges=True,
    )

    transformer = create_coordinate_transformer(
        G_projected.graph["crs"]
    )

    return edges, transformer


# ============================================================
# Load GeoLife trajectory
# ============================================================

def load_trajectory():

    if not TRAJECTORY_FILE.exists():
        raise FileNotFoundError(
            f"Trajectory file not found:\n"
            f"{TRAJECTORY_FILE}"
        )

    df = pd.read_csv(
        TRAJECTORY_FILE
    )

    print(
        f"Loaded trajectory: "
        f"{len(df):,} GPS observations"
    )

    return df


# ============================================================
# Generate candidates
# ============================================================

def generate_candidates(
    trajectory,
    edges,
    transformer,
):

    results = []

    total_points = len(trajectory)

    print("\nGenerating road candidates...")

    for point_number, (_, row) in enumerate(
        trajectory.iterrows(),
        start=1,
    ):

        latitude = row["latitude"]
        longitude = row["longitude"]

        candidates = find_candidates(
            latitude=latitude,
            longitude=longitude,
            edges=edges,
            transformer=transformer,
            radius_meters=SEARCH_RADIUS_METERS,
            max_candidates=MAX_CANDIDATES,
        )

        # Store one row for every candidate.
        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):

            results.append(
                {
                    "user_id": row["user_id"],
                    "trajectory_id": row[
                        "trajectory_id"
                    ],
                    "segment_id": row[
                        "segment_id"
                    ],
                    "point_number": point_number,
                    "timestamp": row[
                        "timestamp"
                    ],
                    "gps_latitude": latitude,
                    "gps_longitude": longitude,
                    "candidate_rank": rank,
                    "u": candidate["u"],
                    "v": candidate["v"],
                    "key": candidate["key"],
                    "distance_meters": candidate[
                        "distance_meters"
                    ],
                    "road_type": candidate[
                        "road_type"
                    ],
                    "road_name": candidate[
                        "road_name"
                    ],
                    "road_length_meters": candidate[
                        "length_meters"
                    ],
                }
            )

        # Progress display
        if (
            point_number % 100 == 0
            or point_number == total_points
        ):

            print(
                f"  Processed "
                f"{point_number:,} / "
                f"{total_points:,} GPS points"
            )

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("TRAJECTORY-LEVEL ROAD CANDIDATE GENERATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    edges, transformer = load_road_network()

    trajectory = load_trajectory()

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidate_df = generate_candidates(
        trajectory,
        edges,
        transformer,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    gps_points = len(
        trajectory
    )

    matched_points = (
        candidate_df[
            "point_number"
        ].nunique()
        if not candidate_df.empty
        else 0
    )

    points_without_candidates = (
        gps_points - matched_points
    )

    candidate_count = len(
        candidate_df
    )

    print("\n" + "=" * 60)
    print("CANDIDATE GENERATION SUMMARY")
    print("=" * 60)

    print(
        f"GPS observations       : "
        f"{gps_points:,}"
    )

    print(
        f"GPS points with candidates: "
        f"{matched_points:,}"
    )

    print(
        f"GPS points without candidates: "
        f"{points_without_candidates:,}"
    )

    print(
        f"Total candidate records : "
        f"{candidate_count:,}"
    )

    if matched_points > 0:

        print(
            f"Average candidates/GPS point: "
            f"{candidate_count / matched_points:.2f}"
        )

    if not candidate_df.empty:

        print("\nCandidate distance statistics:")

        print(
            candidate_df[
                "distance_meters"
            ].describe()
        )

    print("\nSaved to:")

    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print(
        "TRAJECTORY CANDIDATE GENERATION COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()