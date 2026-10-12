
from pathlib import Path
import math

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    PROJECT_ROOT
    / "data"
    / "road_network"
    / "beijing_drive.graphml"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "map_matching"
    / "20081023025304_emission_probabilities.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "map_matching"
    / "20081023025304_transition_probabilities.csv"
)

BETA_METERS = 50.0


def transition_log_score(
    network_distance_meters: float,
    gps_distance_meters: float,
    beta_meters: float = BETA_METERS,
) -> float:
    """Return the project's distance-difference transition log-score."""
    if not math.isfinite(beta_meters) or beta_meters <= 0:
        raise ValueError("Transition beta must be a positive finite number.")
    if not math.isfinite(network_distance_meters):
        return -np.inf
    return -abs(network_distance_meters - gps_distance_meters) / beta_meters


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate GPS-to-GPS distance in metres."""
    radius = 6_371_000.0

    lat1, lon1, lat2, lon2 = map(
        math.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * radius * math.asin(
        min(1.0, math.sqrt(a))
    )


def main():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Road network not found: {GRAPH_FILE}"
        )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Emission file not found: {INPUT_FILE}"
        )

    print("Loading road network and emission data...")

    graph = ox.load_graphml(GRAPH_FILE)
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "user_id",
        "trajectory_id",
        "segment_id",
        "point_number",
        "timestamp",
        "gps_latitude",
        "gps_longitude",
        "u",
        "v",
        "key",
        "emission_probability",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("The candidate CSV is empty.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )

    df = df.reset_index(drop=True)
    df["candidate_id"] = np.arange(len(df))

    # Project the OSM network so geometric distances are in metres.
    projected_graph = ox.project_graph(graph)
    projected_crs = projected_graph.graph["crs"]

    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )

    # Store every directed edge, including its geometry and length.
    edges = {}

    for u, v, key, data in projected_graph.edges(
        keys=True, data=True
    ):
        geometry = data.get("geometry")

        if geometry is None:
            node_u = projected_graph.nodes[u]
            node_v = projected_graph.nodes[v]

            geometry = LineString([
                (node_u["x"], node_u["y"]),
                (node_v["x"], node_v["y"]),
            ])

        edges[(str(u), str(v), str(key))] = {
            "u": u,
            "v": v,
            "key": key,
            "geometry": geometry,
            "length": float(
                data.get("length", geometry.length)
            ),
        }

    # Project each GPS point onto its candidate edge.
    edge_positions = {}

    for _, row in df.iterrows():
        candidate_id = int(row["candidate_id"])

        edge_id = (
            str(row["u"]),
            str(row["v"]),
            str(row["key"]),
        )

        edge = edges.get(edge_id)

        if edge is None:
            continue

        x, y = transformer.transform(
            float(row["gps_longitude"]),
            float(row["gps_latitude"]),
        )

        gps_point = Point(x, y)
        geometry = edge["geometry"]

        if geometry.length > 0:
            fraction = (
                geometry.project(gps_point) / geometry.length
            )
        else:
            fraction = 0.0

        edge_positions[candidate_id] = {
            **edge,
            "fraction": min(1.0, max(0.0, fraction)),
        }

    # Cache shortest-path distances between edge endpoints.
    path_cache = {}

    def get_network_distance(source_id, target_id):
        """Estimate directed road distance between two candidates."""
        source = edge_positions.get(source_id)
        target = edge_positions.get(target_id)

        if source is None or target is None:
            return float("inf")

        source_edge = (
            str(source["u"]),
            str(source["v"]),
            str(source["key"]),
        )

        target_edge = (
            str(target["u"]),
            str(target["v"]),
            str(target["key"]),
        )

        source_fraction = source["fraction"]
        target_fraction = target["fraction"]

        # Both candidates lie on the same directed edge.
        if (
            source_edge == target_edge
            and target_fraction >= source_fraction
        ):
            return (
                target_fraction - source_fraction
            ) * source["length"]

        remaining_source = (
            1.0 - source_fraction
        ) * source["length"]

        partial_target = (
            target_fraction * target["length"]
        )

        cache_key = (source["v"], target["u"])

        if cache_key not in path_cache:
            try:
                path_cache[cache_key] = (
                    nx.shortest_path_length(
                        projected_graph,
                        source=source["v"],
                        target=target["u"],
                        weight="length",
                    )
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                path_cache[cache_key] = float("inf")

        middle_distance = path_cache[cache_key]

        if not math.isfinite(middle_distance):
            return float("inf")

        return (
            remaining_source
            + middle_distance
            + partial_target
        )

    output_rows = []

    segment_columns = [
        "user_id",
        "trajectory_id",
        "segment_id",
    ]

    segments = df.groupby(
        segment_columns,
        sort=False,
        dropna=False,
    )

    print("Calculating candidate transitions...")

    for segment_key, segment in segments:
        segment = segment.sort_values(
            ["point_number", "timestamp"],
            kind="stable",
        )

        # Each point_number represents one GPS observation.
        observations = list(
            segment.groupby("point_number", sort=True)
        )

        for i in range(1, len(observations)):
            previous_point, previous_candidates = observations[i - 1]
            current_point, current_candidates = observations[i]

            previous_candidates = previous_candidates.reset_index(
                drop=True
            )
            current_candidates = current_candidates.reset_index(
                drop=True
            )

            previous_gps = previous_candidates.iloc[0]
            current_gps = current_candidates.iloc[0]

            gps_distance = haversine_distance(
                previous_gps["gps_latitude"],
                previous_gps["gps_longitude"],
                current_gps["gps_latitude"],
                current_gps["gps_longitude"],
            )

            # Normalize transitions separately for each previous
            # candidate, across all reachable current candidates.
            for _, previous in previous_candidates.iterrows():
                previous_id = int(previous["candidate_id"])

                pair_results = []
                log_scores = []

                for _, current in current_candidates.iterrows():
                    current_id = int(current["candidate_id"])

                    road_distance = get_network_distance(
                        previous_id, current_id
                    )

                    log_score = transition_log_score(
                        road_distance, gps_distance, BETA_METERS
                    )

                    pair_results.append({
                        "user_id": segment_key[0],
                        "trajectory_id": segment_key[1],
                        "segment_id": segment_key[2],
                        "previous_point_number": previous_point,
                        "current_point_number": current_point,
                        "previous_candidate_id": previous_id,
                        "current_candidate_id": current_id,
                        "gps_distance_meters": gps_distance,
                        "network_distance_meters": (
                            road_distance
                            if math.isfinite(road_distance)
                            else np.nan
                        ),
                        "transition_log_score": log_score,
                    })

                    log_scores.append(log_score)

                log_scores = np.asarray(log_scores, dtype=float)
                probabilities = np.zeros(len(log_scores))

                reachable = np.isfinite(log_scores)

                if reachable.any():
                    shifted = (
                        log_scores[reachable]
                        - np.max(log_scores[reachable])
                    )

                    weights = np.exp(shifted)
                    probabilities[reachable] = (
                        weights / weights.sum()
                    )

                for result, probability in zip(
                    pair_results, probabilities
                ):
                    result["transition_probability"] = probability
                    output_rows.append(result)

    result_df = pd.DataFrame(output_rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\nTransition model completed successfully.")
    print(f"Input candidate records: {len(df)}")
    print(f"Candidate transitions: {len(result_df)}")
    print(f"Beta: {BETA_METERS} metres")
    print(f"Output saved to: {OUTPUT_FILE}")

    if not result_df.empty:
        print("\nSample transitions:")
        print(
            result_df[
                [
                    "previous_candidate_id",
                    "current_candidate_id",
                    "gps_distance_meters",
                    "network_distance_meters",
                    "transition_probability",
                ]
            ].head(10).to_string(index=False)
        )


if __name__ == "__main__":
    main()
