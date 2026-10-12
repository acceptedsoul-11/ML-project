"""Reusable HMM/Viterbi inference over the project's saved Beijing OSM graph."""

from __future__ import annotations

import logging
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, Point

from src.map_matching.candidate_generation import (
    GRAPH_FILE,
    create_coordinate_transformer,
    find_candidates,
)
from src.map_matching.hmm_transition import (
    BETA_METERS,
    haversine_distance,
    transition_log_score,
)
from src.map_matching.hmm_emission import (
    GPS_SIGMA_METERS,
    calculate_emission_probabilities,
)
from src.map_matching.viterbi import viterbi_decode

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RADIUS_METERS = 100.0
DEFAULT_MAX_CANDIDATES = 5
OBSERVATION_COLUMNS = ("latitude", "longitude")


def validate_coordinates(latitude: Any, longitude: Any) -> tuple[float, float]:
    """Return validated WGS84 coordinates or raise a readable ValueError."""
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("Latitude and longitude must be numeric values.") from exc
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("Latitude and longitude must be finite numbers.")
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    return lat, lon


@lru_cache(maxsize=1)
def load_road_network() -> tuple[nx.MultiDiGraph, Any, Transformer, Transformer]:
    """Load and project the saved OSM graph once per Python process."""
    if not GRAPH_FILE.is_file():
        raise FileNotFoundError(f"Saved OSM graph not found: {GRAPH_FILE}")
    graph = ox.load_graphml(GRAPH_FILE)
    projected = ox.project_graph(graph)
    edges = ox.graph_to_gdfs(projected, nodes=False, edges=True)
    to_projected = create_coordinate_transformer(projected.graph["crs"])
    to_wgs84 = Transformer.from_crs(
        projected.graph["crs"], "EPSG:4326", always_xy=True
    )
    LOGGER.info("Loaded road graph: %s nodes, %s directed edges", len(graph), len(edges))
    return projected, edges, to_projected, to_wgs84


def _display_value(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, set)):
        return ", ".join(map(str, value))
    return str(value)


def _candidate_frame(
    latitude: float,
    longitude: float,
    edges: Any,
    transformer: Transformer,
    radius_meters: float,
    max_candidates: int,
    gps_sigma_meters: float,
) -> pd.DataFrame:
    candidates = find_candidates(
        latitude=latitude,
        longitude=longitude,
        edges=edges,
        transformer=transformer,
        radius_meters=radius_meters,
        max_candidates=max_candidates,
    )
    if not candidates:
        return pd.DataFrame()
    frame = pd.DataFrame(candidates)
    frame = frame.drop_duplicates(["u", "v", "key"], keep="first").reset_index(drop=True)
    frame["candidate_rank"] = np.arange(1, len(frame) + 1)
    frame["_observation_id"] = 0
    frame = calculate_emission_probabilities(
        frame, gps_sigma_meters, ["_observation_id"]
    ).drop(columns="_observation_id")
    frame["road_name"] = frame["road_name"].map(_display_value)
    frame["road_type"] = frame["road_type"].map(_display_value)
    if "maxspeed" in frame:
        frame["maxspeed"] = frame["maxspeed"].map(_display_value)
    return frame


def predict_single_point(
    latitude: Any,
    longitude: Any,
    *,
    radius_meters: float = DEFAULT_RADIUS_METERS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    gps_sigma_meters: float = GPS_SIGMA_METERS,
    graph_context: tuple[nx.MultiDiGraph, Any, Transformer, Transformer] | None = None,
) -> pd.DataFrame:
    """Score nearby directed OSM edges for one point; this is not route-level HMM."""
    lat, lon = validate_coordinates(latitude, longitude)
    if radius_meters <= 0 or max_candidates < 1 or gps_sigma_meters <= 0:
        raise ValueError("Search radius, candidate count, and GPS sigma must be positive.")
    _, edges, transformer, _ = graph_context or load_road_network()
    result = _candidate_frame(
        lat, lon, edges, transformer, radius_meters, max_candidates, gps_sigma_meters
    )
    if not result.empty:
        result["gps_latitude"] = lat
        result["gps_longitude"] = lon
    return result


_COLUMN_ALIASES = {
    "latitude": ("latitude", "lat", "gps_latitude", "gps_lat"),
    "longitude": ("longitude", "lon", "lng", "gps_longitude", "gps_lon"),
    "timestamp": ("timestamp", "time", "datetime", "date_time"),
    "segment_id": ("segment_id", "segment", "trip_segment"),
    "speed_kmh": ("speed_kmh", "speed_kph"),
    "heading_deg": ("heading_deg", "heading", "course", "bearing"),
}


def normalize_trajectory(
    data: pd.DataFrame,
    *,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    timestamp_column: str | None = None,
    segment_column: str | None = None,
) -> pd.DataFrame:
    """Validate and normalize a trajectory without inventing observations."""
    if data.empty:
        raise ValueError("The trajectory contains no rows.")
    columns = {str(column).strip().lower(): column for column in data.columns}

    def resolve(explicit: str | None, aliases: tuple[str, ...], label: str) -> Any:
        if explicit:
            if explicit not in data.columns:
                raise ValueError(f"Selected {label} column '{explicit}' does not exist.")
            return explicit
        for alias in aliases:
            if alias in columns:
                return columns[alias]
        if label in ("latitude", "longitude"):
            raise ValueError(
                f"Could not identify a {label} column. Available columns: "
                + ", ".join(map(str, data.columns))
            )
        return None

    lat_col = resolve(latitude_column, _COLUMN_ALIASES["latitude"], "latitude")
    lon_col = resolve(longitude_column, _COLUMN_ALIASES["longitude"], "longitude")
    if lat_col == lon_col:
        raise ValueError("Latitude and longitude must use different columns.")
    time_col = resolve(timestamp_column, _COLUMN_ALIASES["timestamp"], "timestamp")
    seg_col = resolve(segment_column, _COLUMN_ALIASES["segment_id"], "segment")

    result = pd.DataFrame(index=data.index)
    result["point_number"] = np.arange(1, len(data) + 1)
    result["latitude"] = pd.to_numeric(data[lat_col], errors="coerce")
    result["longitude"] = pd.to_numeric(data[lon_col], errors="coerce")
    invalid = result[["latitude", "longitude"]].isna().any(axis=1)
    if invalid.any():
        row_numbers = (np.flatnonzero(invalid.to_numpy()) + 2).tolist()[:8]
        raise ValueError(
            "Latitude/longitude must be numeric and present on every row. "
            f"Invalid CSV row(s): {row_numbers}"
        )
    if ((result.latitude < -90) | (result.latitude > 90)).any():
        raise ValueError("Latitude values must be between -90 and 90 degrees.")
    if ((result.longitude < -180) | (result.longitude > 180)).any():
        raise ValueError("Longitude values must be between -180 and 180 degrees.")
    if time_col is not None:
        result["timestamp"] = data[time_col].astype("string").fillna("")
    else:
        result["timestamp"] = ""
    if seg_col is not None:
        if data[seg_col].isna().any():
            raise ValueError("Segment values must be present for every observation.")
        result["segment_id"] = data[seg_col].astype(str)
    else:
        result["segment_id"] = "0"
    for feature in ("speed_kmh", "heading_deg"):
        source_column = resolve(None, _COLUMN_ALIASES[feature], feature)
        if source_column is not None:
            result[feature] = pd.to_numeric(data[source_column], errors="coerce")
    result = result.reset_index(drop=True)
    return result


def _edge_geometry(graph: nx.MultiDiGraph, edge: pd.Series) -> LineString:
    data = graph.get_edge_data(edge["u"], edge["v"], edge["key"])
    if data is None:
        raise ValueError("A candidate edge is missing from the saved road graph.")
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry
    return LineString([
        (graph.nodes[edge["u"]]["x"], graph.nodes[edge["u"]]["y"]),
        (graph.nodes[edge["v"]]["x"], graph.nodes[edge["v"]]["y"]),
    ])


def _road_distance(
    graph: nx.MultiDiGraph,
    source: pd.Series,
    target: pd.Series,
    projected_gps: tuple[Point, Point],
    edge_geometry_cache: dict[tuple[Any, Any, Any], LineString],
    dijkstra_cache: dict[Any, dict[Any, float]],
) -> float:
    source_id = (source.u, source.v, source.key)
    target_id = (target.u, target.v, target.key)
    source_geometry = edge_geometry_cache.setdefault(source_id, _edge_geometry(graph, source))
    target_geometry = edge_geometry_cache.setdefault(target_id, _edge_geometry(graph, target))
    source_fraction = source_geometry.project(projected_gps[0]) / source_geometry.length if source_geometry.length else 0.0
    target_fraction = target_geometry.project(projected_gps[1]) / target_geometry.length if target_geometry.length else 0.0
    source_fraction = min(1.0, max(0.0, source_fraction))
    target_fraction = min(1.0, max(0.0, target_fraction))
    if source_id == target_id and target_fraction >= source_fraction:
        return (target_fraction - source_fraction) * float(source.get("length", source_geometry.length))
    source_length = float(source.get("length", source_geometry.length))
    target_length = float(target.get("length", target_geometry.length))
    source_node, target_node = source.v, target.u
    if source_node not in dijkstra_cache:
        try:
            dijkstra_cache[source_node] = nx.single_source_dijkstra_path_length(
                graph, source_node, weight="length"
            )
        except nx.NodeNotFound:
            dijkstra_cache[source_node] = {}
    middle = dijkstra_cache[source_node].get(target_node, math.inf)
    if not math.isfinite(middle):
        return math.inf
    return (1 - source_fraction) * source_length + middle + target_fraction * target_length


def _decode_segment(
    observations: list[pd.DataFrame],
    projected_points: dict[int, Point],
    graph: nx.MultiDiGraph,
    edge_geometry_cache: dict[tuple[Any, Any, Any], LineString],
    dijkstra_cache: dict[Any, dict[Any, float]],
) -> list[dict[str, Any]]:
    """Viterbi decode each connected run, restarting at unreachable transitions."""
    output: list[dict[str, Any]] = []
    block_number = 0
    run: list[pd.DataFrame] = []
    run_transitions: list[np.ndarray] = []
    run_scores: np.ndarray | None = None

    def flush() -> None:
        nonlocal block_number, run, run_transitions, run_scores
        if not run:
            return
        block_number += 1
        emissions = [observation["emission_log_score"].to_numpy(dtype=float) for observation in run]
        selected_indices, score_history, best_final_score = viterbi_decode(emissions, run_transitions)
        if selected_indices is None:
            raise RuntimeError("A connected Viterbi block did not produce a path.")
        for offset, (observation, index) in enumerate(zip(run, selected_indices)):
            row = observation.iloc[index].to_dict()
            row["sequence_block"] = block_number
            row["matched"] = True
            row["viterbi_cumulative_score"] = score_history[offset][index]
            row["viterbi_best_path_score"] = best_final_score
            output.append(row)
        run, run_transitions, run_scores = [], [], None

    for current in observations:
        if current.empty:
            flush()
            continue
        if not run:
            run = [current]
            run_scores = current["emission_log_score"].to_numpy(dtype=float)
            continue
        previous = run[-1]
        previous_gps = previous.iloc[0]
        current_gps = current.iloc[0]
        gps_distance = haversine_distance(
            float(previous_gps.gps_latitude), float(previous_gps.gps_longitude),
            float(current_gps.gps_latitude), float(current_gps.gps_longitude),
        )
        transition = np.full((len(previous), len(current)), -np.inf, dtype=float)
        for i, (_, source) in enumerate(previous.iterrows()):
            for j, (_, target) in enumerate(current.iterrows()):
                network_distance = _road_distance(
                    graph, source, target,
                    (projected_points[int(previous.iloc[0].point_number)], projected_points[int(current.iloc[0].point_number)]),
                    edge_geometry_cache, dijkstra_cache,
                )
                if math.isfinite(network_distance):
                    transition[i, j] = transition_log_score(network_distance, gps_distance)
        next_scores = np.max(run_scores[:, None] + transition, axis=0)
        next_scores += current["emission_log_score"].to_numpy(dtype=float)
        if not np.isfinite(next_scores).any():
            # Unreachable candidates cannot be joined by a road path. End the prior
            # run so the two sequences are not drawn as if a route connected them.
            flush()
            run = [current]
            run_scores = current["emission_log_score"].to_numpy(dtype=float)
            continue
        run_transitions.append(transition)
        run.append(current)
        run_scores = next_scores
    flush()
    return output


def generate_trajectory_candidates(
    data: pd.DataFrame,
    *,
    radius_meters: float = DEFAULT_RADIUS_METERS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    gps_sigma_meters: float = GPS_SIGMA_METERS,
    graph_context: tuple[nx.MultiDiGraph, Any, Transformer, Transformer] | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    timestamp_column: str | None = None,
    segment_column: str | None = None,
) -> dict[str, Any]:
    """Validate a trajectory and generate projected road candidates once."""
    started = time.perf_counter()
    if radius_meters <= 0 or max_candidates < 1 or gps_sigma_meters <= 0:
        raise ValueError("Search radius, candidate count, and GPS sigma must be positive.")
    trajectory = normalize_trajectory(
        data,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        timestamp_column=timestamp_column,
        segment_column=segment_column,
    )
    graph, edges, to_projected, to_wgs84 = graph_context or load_road_network()
    all_candidates: list[pd.DataFrame] = []
    obs_candidates: dict[int, pd.DataFrame] = {}
    projected_points: dict[int, Point] = {}
    for row in trajectory.itertuples(index=False):
        point_number = int(row.point_number)
        x, y = to_projected.transform(float(row.longitude), float(row.latitude))
        gps_point = Point(x, y)
        projected_points[point_number] = gps_point
        candidates = _candidate_frame(
            float(row.latitude), float(row.longitude), edges, to_projected,
            radius_meters, max_candidates, gps_sigma_meters,
        )
        if not candidates.empty:
            for col in ("u", "v", "key"):
                candidates[col] = candidates[col].map(
                    lambda value: int(value) if isinstance(value, str) and value.isdigit() else value
                )
            candidates["point_number"] = point_number
            candidates["timestamp"] = row.timestamp
            candidates["segment_id"] = row.segment_id
            candidates["gps_latitude"] = row.latitude
            candidates["gps_longitude"] = row.longitude
            if hasattr(row, "speed_kmh"):
                candidates["speed_kmh"] = row.speed_kmh
            if hasattr(row, "heading_deg"):
                candidates["heading_deg"] = row.heading_deg
            obs_candidates[point_number] = candidates
            all_candidates.append(candidates)
    candidate_frame = pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame()
    return {
        "trajectory": trajectory,
        "candidates": candidate_frame,
        "observation_candidates": obs_candidates,
        "projected_points": projected_points,
        "graph": graph,
        "edges": edges,
        "to_projected": to_projected,
        "to_wgs84": to_wgs84,
        "preprocessing_seconds": time.perf_counter() - started,
    }


def predict_trajectory(
    data: pd.DataFrame,
    *,
    radius_meters: float = DEFAULT_RADIUS_METERS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    gps_sigma_meters: float = GPS_SIGMA_METERS,
    graph_context: tuple[nx.MultiDiGraph, Any, Transformer, Transformer] | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    timestamp_column: str | None = None,
    segment_column: str | None = None,
) -> dict[str, Any]:
    """Generate candidates and run the existing network-distance Viterbi baseline."""
    prepared = generate_trajectory_candidates(
        data,
        radius_meters=radius_meters,
        max_candidates=max_candidates,
        gps_sigma_meters=gps_sigma_meters,
        graph_context=graph_context,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        timestamp_column=timestamp_column,
        segment_column=segment_column,
    )
    return decode_network_hmm(prepared)


def decode_network_hmm(prepared: dict[str, Any]) -> dict[str, Any]:
    """Run the existing directed-network full-sequence HMM on prepared candidates."""
    started = time.perf_counter()
    trajectory = prepared["trajectory"]
    graph = prepared["graph"]
    obs_candidates = prepared["observation_candidates"]
    projected_points = prepared["projected_points"]
    candidate_frame = prepared["candidates"]
    results: list[dict[str, Any]] = []
    edge_geometry_cache: dict[tuple[Any, Any, Any], LineString] = {}
    dijkstra_cache: dict[Any, dict[Any, float]] = {}
    for segment_id, segment in trajectory.groupby("segment_id", sort=False, dropna=False):
        observations: list[pd.DataFrame] = []
        for row in segment.itertuples(index=False):
            candidates = obs_candidates.get(int(row.point_number))
            if candidates is None:
                observations.append(pd.DataFrame())
            else:
                candidates = candidates.copy()
                observations.append(candidates)
        decoded = _decode_segment(
            observations, projected_points, graph, edge_geometry_cache, dijkstra_cache
        )
        results.extend(decoded)
    selected = pd.DataFrame(results)
    if not selected.empty:
        selected = selected.drop_duplicates("point_number", keep="first")
        selected = selected.set_index("point_number")
    output = trajectory.copy().set_index("point_number", drop=False)
    output["matched"] = False
    output["sequence_block"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["u"] = pd.NA
    output["v"] = pd.NA
    output["key"] = pd.NA
    for column in ("road_name", "road_type"):
        output[column] = pd.NA
    output["distance_meters"] = np.nan
    output["emission_probability"] = np.nan
    if not selected.empty:
        for index, row in selected.iterrows():
            for col in ("matched", "sequence_block", "u", "v", "key", "road_name", "road_type", "distance_meters", "emission_probability"):
                output.at[index, col] = row.get(col)
    output = output.reset_index(drop=True)
    if not candidate_frame.empty:
        candidate_frame = candidate_frame.drop(columns=[], errors="ignore")
    decoding_seconds = time.perf_counter() - started
    elapsed = prepared["preprocessing_seconds"] + decoding_seconds
    LOGGER.info("Matched %s of %s observations in %.2fs", int(output.matched.sum()), len(output), elapsed)
    return {
        "observations": output,
        "candidates": candidate_frame,
        "runtime_seconds": elapsed,
        "preprocessing_seconds": prepared["preprocessing_seconds"],
        "decoding_seconds": decoding_seconds,
        "graph": graph,
        "edges": prepared["edges"],
        "to_wgs84": prepared["to_wgs84"],
    }

