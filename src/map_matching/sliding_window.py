"""Paper-informed naive-HMM scoring and delayed sliding-window Viterbi."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from src.map_matching.inference import _edge_geometry
from src.map_matching.viterbi import viterbi_decode


@dataclass(frozen=True)
class FeatureWeights:
    """Untrained feature mixture; supplied paper's learned weights are unavailable."""

    emission_distance: float = 0.70
    emission_heading: float = 0.20
    emission_speed_limit: float = 0.10
    transition_backtrack: float = 0.25
    transition_midpoint: float = 0.75

    def validate(self) -> None:
        values = tuple(self.__dict__.values())
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Feature weights must be finite and non-negative.")
        if self.emission_distance + self.emission_heading + self.emission_speed_limit <= 0:
            raise ValueError("At least one emission weight must be positive.")
        if self.transition_backtrack + self.transition_midpoint <= 0:
            raise ValueError("At least one transition weight must be positive.")


DEFAULT_FEATURE_WEIGHTS = FeatureWeights()


def validate_window(window_size: int, buffer_size: int) -> tuple[int, int]:
    """Validate T and N, where T is the maximum window and N future points."""
    if isinstance(window_size, bool) or int(window_size) != window_size or window_size < 1:
        raise ValueError("Window size T must be a positive integer.")
    if isinstance(buffer_size, bool) or int(buffer_size) != buffer_size or buffer_size < 0:
        raise ValueError("Buffer size N must be a non-negative integer.")
    if buffer_size >= window_size:
        raise ValueError("Buffer size N must be smaller than window size T.")
    return int(window_size), int(buffer_size)


def parse_speed_limit_kmh(value: Any) -> float | None:
    """Parse a numeric OSM speed limit; unitless OSM values are treated as km/h."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple, set)):
        value = next(iter(value), None)
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    speed = float(match.group())
    if "mph" in str(value).lower():
        speed *= 1.609344
    return speed if math.isfinite(speed) and speed > 0 else None


def _edge_key(row: pd.Series | Any) -> tuple[Any, Any, Any]:
    def normalize(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return value
    return tuple(normalize(row[name]) for name in ("u", "v", "key"))


def _bearing_degrees(a: tuple[float, float], b: tuple[float, float]) -> float | None:
    dx, dy = b[0] - a[0], b[1] - a[1]
    if math.hypot(dx, dy) < 0.5:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360


def _observation_headings(
    segment: pd.DataFrame, transformer: Transformer
) -> dict[int, float | None]:
    """Use supplied course or only the previous GPS point, avoiding future leakage."""
    result: dict[int, float | None] = {}
    previous_projected: tuple[float, float] | None = None
    for row in segment.itertuples(index=False):
        index = int(row.point_number)
        supplied = getattr(row, "heading_deg", np.nan)
        if pd.notna(supplied) and math.isfinite(float(supplied)):
            result[index] = float(supplied) % 360
        else:
            x, y = transformer.transform(float(row.longitude), float(row.latitude))
            current = (x, y)
            result[index] = _bearing_degrees(previous_projected, current) if previous_projected else None
        x, y = transformer.transform(float(row.longitude), float(row.latitude))
        previous_projected = (x, y)
    return result


def _candidate_state(
    observation: pd.Series,
    candidates: pd.DataFrame,
    graph: Any,
    weights: FeatureWeights,
    gps_sigma_meters: float,
    heading: float | None,
) -> dict[str, Any]:
    candidates = candidates.reset_index(drop=True).copy()
    emissions: list[float] = []
    midpoints: list[tuple[float, float]] = []
    edge_ids: list[tuple[Any, Any, Any]] = []
    for _, candidate in candidates.iterrows():
        edge_id = _edge_key(candidate)
        geometry = _edge_geometry(graph, candidate)
        midpoint = geometry.interpolate(0.5, normalized=True)
        midpoints.append((float(midpoint.x), float(midpoint.y)))
        edge_ids.append(edge_id)

    speed = observation.get("speed_kmh", np.nan)
    speed = float(speed) if pd.notna(speed) and math.isfinite(float(speed)) else None
    feature_values = []
    for index, (_, candidate) in enumerate(candidates.iterrows()):
        values = [(weights.emission_distance, math.exp(-float(candidate.distance_meters) ** 2 / (2 * gps_sigma_meters**2)))]
        if heading is not None:
            geometry = _edge_geometry(graph, candidate)
            coords = list(geometry.coords)
            road_heading = _bearing_degrees(coords[0], coords[-1]) if len(coords) >= 2 else None
            if road_heading is not None:
                difference = math.radians(road_heading - heading)
                alignment = (1 + math.cos(difference)) / 2
                values.append((weights.emission_heading, alignment))
        speed_limit = parse_speed_limit_kmh(candidate.get("maxspeed"))
        if speed is not None and speed_limit is not None:
            mph = speed / 1.609344
            consistent = not (mph > 40 and speed >= 1.2 * speed_limit)
            values.append((weights.emission_speed_limit, float(consistent)))
        weight_sum = sum(weight for weight, _ in values)
        score = sum(weight * value for weight, value in values) / weight_sum if weight_sum > 0 else 0.0
        feature_values.append(max(score, 1e-300))
    probabilities = np.asarray(feature_values, dtype=float)
    probabilities /= probabilities.sum()
    emissions = np.log(np.maximum(probabilities, 1e-300))
    return {
        "observation": observation,
        "candidates": candidates,
        "edge_ids": edge_ids,
        "midpoints": midpoints,
        "emissions": emissions,
    }


def _transition_log_matrix(
    previous: dict[str, Any],
    current: dict[str, Any],
    visited: set[tuple[Any, Any, Any]],
    weights: FeatureWeights,
    gps_distance_meters: float,
) -> np.ndarray:
    """Combine paper-inspired features without requiring graph shortest paths."""
    matrix = np.full((len(previous["edge_ids"]), len(current["edge_ids"])), -np.inf)
    for i, source_edge in enumerate(previous["edge_ids"]):
        target_scores = []
        for j, target_edge in enumerate(current["edge_ids"]):
            midpoint_a, midpoint_b = previous["midpoints"][i], current["midpoints"][j]
            midpoint_distance = math.hypot(midpoint_b[0] - midpoint_a[0], midpoint_b[1] - midpoint_a[1])
            discrepancy = 1.0 / (1.0 + abs(gps_distance_meters - midpoint_distance))
            backtrack = float(target_edge not in visited or target_edge == source_edge)
            total_weight = weights.transition_backtrack + weights.transition_midpoint
            score = (
                weights.transition_backtrack * backtrack
                + weights.transition_midpoint * discrepancy
            ) / total_weight
            target_scores.append(max(score, 1e-300))
        probabilities = np.asarray(target_scores, dtype=float)
        probabilities /= probabilities.sum()
        matrix[i, :] = np.log(np.maximum(probabilities, 1e-300))
    return matrix


def _gps_distance(a: pd.Series, b: pd.Series) -> float:
    from src.map_matching.hmm_transition import haversine_distance
    return haversine_distance(float(a.latitude), float(a.longitude), float(b.latitude), float(b.longitude))


def run_sliding_window_viterbi(
    prepared: dict[str, Any],
    *,
    window_size: int,
    buffer_size: int,
    gps_sigma_meters: float = 20.0,
    weights: FeatureWeights = DEFAULT_FEATURE_WEIGHTS,
) -> dict[str, Any]:
    """Decode with a trailing T-observation window and N-point delayed commitment.

    On arrival at index t, the current best window path finalizes through t-N.
    At each segment end the remaining tail is flushed with the observations seen.
    """
    window_size, buffer_size = validate_window(window_size, buffer_size)
    weights.validate()
    if gps_sigma_meters <= 0 or not math.isfinite(gps_sigma_meters):
        raise ValueError("GPS sigma must be a positive finite value.")
    trajectory: pd.DataFrame = prepared["trajectory"]
    candidates: pd.DataFrame = prepared["candidates"]
    graph, transformer = prepared["graph"], prepared["to_projected"]
    started = __import__("time").perf_counter()
    records: list[dict[str, Any]] = []
    block_number = 0

    for segment_id, segment in trajectory.groupby("segment_id", sort=False, dropna=False):
        segment = segment.sort_values("point_number", kind="stable").reset_index(drop=True)
        headings = _observation_headings(segment, transformer)
        states: list[dict[str, Any] | None] = []
        for row in segment.to_dict("records"):
            point_number = int(row["point_number"])
            point_candidates = candidates[candidates.point_number == point_number] if not candidates.empty else pd.DataFrame()
            observation = pd.Series(row)
            states.append(
                _candidate_state(observation, point_candidates, graph, weights, gps_sigma_meters, headings[point_number])
                if not point_candidates.empty else None
            )

        chosen: dict[int, pd.Series] = {}
        finalized_at: dict[int, int] = {}
        committed_blocks: dict[int, int] = {}
        visited: set[tuple[Any, Any, Any]] = set()
        next_uncommitted = 0
        run_start: int | None = None
        run_block = 0

        def decode_window(start: int, end: int) -> tuple[list[int], list[dict[str, Any]]]:
            window_states: list[dict[str, Any]] = []
            local_positions: list[int] = []
            for local in range(start, end + 1):
                state = states[local]
                if state is None:
                    raise ValueError("A window cannot cross an observation without road candidates.")
                if local < next_uncommitted:
                    fixed = chosen[local]
                    edge_id = _edge_key(fixed)
                    position = next((i for i, key in enumerate(state["edge_ids"]) if key == edge_id), None)
                    if position is None:
                        raise RuntimeError("A previously committed road disappeared from the candidate window.")
                    frozen = dict(state)
                    frozen["candidates"] = state["candidates"].iloc[[position]].reset_index(drop=True)
                    frozen["edge_ids"] = [state["edge_ids"][position]]
                    frozen["midpoints"] = [state["midpoints"][position]]
                    frozen["emissions"] = state["emissions"][position:position + 1]
                    state = frozen
                window_states.append(state)
                local_positions.append(local)

            emissions = [state["emissions"] for state in window_states]
            transitions: list[np.ndarray] = []
            first_state_index = local_positions[0]
            if first_state_index > 0:
                previous_row = chosen.get(first_state_index - 1)
                previous_state = states[first_state_index - 1]
                if previous_row is not None and previous_state is not None:
                    prior_edge = _edge_key(previous_row)
                    pidx = next(i for i, key in enumerate(previous_state["edge_ids"]) if key == prior_edge)
                    prior_single = dict(previous_state)
                    prior_single["edge_ids"] = [prior_edge]
                    prior_single["midpoints"] = [previous_state["midpoints"][pidx]]
                    current_row = segment.iloc[first_state_index]
                    prior_row = segment.iloc[first_state_index - 1]
                    transitions.append(_transition_log_matrix(
                        prior_single, window_states[0], visited | {prior_edge}, weights,
                        _gps_distance(prior_row, current_row),
                    ))
                    emissions[0] = emissions[0] + transitions.pop(0)[0]

            for offset in range(1, len(window_states)):
                prev_idx, curr_idx = local_positions[offset - 1], local_positions[offset]
                prev_row, curr_row = segment.iloc[prev_idx], segment.iloc[curr_idx]
                transitions.append(_transition_log_matrix(
                    window_states[offset - 1], window_states[offset], visited,
                    weights, _gps_distance(prev_row, curr_row),
                ))
            path, _, _ = viterbi_decode(emissions, transitions)
            if path is None:
                raise RuntimeError("The naive-HMM window produced no finite Viterbi path.")
            return path, window_states

        def finalize(commit_through: int, arrival_index: int) -> None:
            nonlocal next_uncommitted
            if run_start is None or commit_through < next_uncommitted:
                return
            start = max(run_start, arrival_index - window_size + 1)
            if next_uncommitted < start:
                raise RuntimeError("Sliding-window buffer exceeded the retained window.")
            path, window_states = decode_window(start, arrival_index)
            commit_through = min(commit_through, arrival_index)
            for local in range(next_uncommitted, commit_through + 1):
                path_offset = local - start
                candidate_row = window_states[path_offset]["candidates"].iloc[path[path_offset]].copy()
                chosen[local] = candidate_row
                finalized_at[local] = arrival_index
                committed_blocks[local] = run_block
                visited.add(_edge_key(candidate_row))
            next_uncommitted = commit_through + 1

        for local, state in enumerate(states):
            if state is None:
                if run_start is not None:
                    finalize(local - 1, local - 1)
                    run_start = None
                next_uncommitted = local + 1
                continue
            if run_start is None:
                run_start = local
                block_number += 1
                run_block = block_number
                next_uncommitted = local
            commit_through = local - buffer_size
            if commit_through >= run_start and commit_through >= next_uncommitted:
                finalize(commit_through, local)

        if run_start is not None:
            finalize(len(states) - 1, len(states) - 1)

        # Emit selected edge rows and delay fields aligned to every input observation.
        parsed_timestamps = pd.to_datetime(segment.timestamp, errors="coerce", utc=True)
        for local, observation in segment.iterrows():
            point_number = int(observation.point_number)
            selected = chosen.get(local)
            row = observation.to_dict()
            row.update({
                "matched": selected is not None,
                "sequence_block": committed_blocks.get(local, pd.NA),
                "u": selected.get("u") if selected is not None else pd.NA,
                "v": selected.get("v") if selected is not None else pd.NA,
                "key": selected.get("key") if selected is not None else pd.NA,
                "road_name": selected.get("road_name") if selected is not None else pd.NA,
                "road_type": selected.get("road_type") if selected is not None else pd.NA,
                "distance_meters": selected.get("distance_meters", np.nan) if selected is not None else np.nan,
                "emission_probability": selected.get("emission_probability", np.nan) if selected is not None else np.nan,
                "finalized_at_point": int(segment.iloc[finalized_at[local]].point_number) if local in finalized_at else point_number,
                "prediction_delay_observations": max(0, finalized_at[local] - local) if local in finalized_at else 0,
                "prediction_delay_seconds": np.nan,
            })
            if local in finalized_at and pd.notna(parsed_timestamps.iloc[local]) and pd.notna(parsed_timestamps.iloc[finalized_at[local]]):
                row["prediction_delay_seconds"] = max(0.0, (parsed_timestamps.iloc[finalized_at[local]] - parsed_timestamps.iloc[local]).total_seconds())
            records.append(row)

    output = pd.DataFrame(records).sort_values("point_number", kind="stable").reset_index(drop=True)
    return {
        "observations": output,
        "decoding_seconds": __import__("time").perf_counter() - started,
        "window_size": window_size,
        "buffer_size": buffer_size,
        "feature_weights": weights,
    }


def nearest_candidate_predictions(prepared: dict[str, Any]) -> pd.DataFrame:
    """Return the nearest candidate per observation as a geometric baseline."""
    trajectory: pd.DataFrame = prepared["trajectory"]
    candidates: pd.DataFrame = prepared["candidates"]
    first = candidates.sort_values(["point_number", "distance_meters"], kind="stable").drop_duplicates("point_number") if not candidates.empty else pd.DataFrame()
    selected = first.set_index("point_number") if not first.empty else pd.DataFrame()
    output = trajectory.copy()
    output["matched"] = output.point_number.isin(selected.index)
    output["sequence_block"] = output.segment_id
    for col in ("u", "v", "key", "road_name", "road_type", "distance_meters", "emission_probability"):
        output[col] = output.point_number.map(selected[col]) if not selected.empty and col in selected else np.nan
    output["prediction_delay_observations"] = 0
    output["prediction_delay_seconds"] = 0.0
    output["finalized_at_point"] = output.point_number
    return output
