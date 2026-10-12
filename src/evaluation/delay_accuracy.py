"""Reproducible baselines and buffer sweeps for map-matching trajectories."""

from __future__ import annotations

import ast
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.map_matching.inference import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RADIUS_METERS,
    GPS_SIGMA_METERS,
    decode_network_hmm,
    generate_trajectory_candidates,
)
from src.map_matching.sliding_window import (
    DEFAULT_FEATURE_WEIGHTS,
    FeatureWeights,
    nearest_candidate_predictions,
    run_sliding_window_viterbi,
    validate_window,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "results" / "experiments"

METRIC_DEFINITIONS = {
    "coverage_percent": "matched observations / input observations * 100. Not accuracy.",
    "ground_truth_accuracy_percent": "Exact selected directed edge match / usable labeled observations * 100. Blank when verified edge labels are unavailable.",
    "gps_to_road_distance_m": "Projected geometric distance from GPS observation to selected edge. A fit proxy, not ground-truth accuracy.",
    "prediction_delay_observations": "For each matched point, number of later observations available when its road prediction was committed. Offline baselines are measured at trajectory end.",
    "prediction_delay_seconds": "Timestamp at commitment minus observation timestamp, when both parse and are ordered.",
    "invalid_transition_count": "Adjacent matched observations with no directed road path from the previous selected edge's downstream node to the next selected edge's upstream node. The naive-HMM model does not enforce this constraint.",
    "runtime_seconds": "Shared candidate preparation time plus the measured decode/selection time for this method.",
}

MODEL_NEAREST = "nearest_road"
MODEL_FULL = "full_sequence_network_hmm"
MODEL_SLIDING = "paper_informed_sliding_window"


def _edge_tuple(row: Any) -> tuple[Any, Any, Any] | None:
    if not bool(getattr(row, "matched", False)):
        return None
    values = []
    for name in ("u", "v", "key"):
        value = getattr(row, name, None)
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        values.append(value)
    return tuple(values)


def _canonical_label(value: Any) -> tuple[str, str, str] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (tuple, list)) and len(value) == 3:
        parts = value
    else:
        raw = str(value).strip()
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (tuple, list)) and len(parsed) == 3:
            parts = parsed
        else:
            parts = raw.strip("()[] ").replace("|", ",").split(",")
            if len(parts) != 3:
                return None
    return tuple(str(part).strip().strip("'\"") for part in parts)


def ground_truth_labels(data: pd.DataFrame, column: str | None = None) -> tuple[dict[int, tuple[str, str, str]], int, str | None]:
    """Read only explicitly edge-shaped labels; never infer truth from GPS or OSM."""
    edge_cols = next((cols for cols in (
        ("ground_truth_u", "ground_truth_v", "ground_truth_key"),
        ("true_u", "true_v", "true_key"),
        ("reference_u", "reference_v", "reference_key"),
    ) if all(col in data.columns for col in cols)), None)
    labels: dict[int, tuple[str, str, str]] = {}
    source = None
    if edge_cols:
        source = ",".join(edge_cols)
        raw_values = [tuple(row) for row in data[list(edge_cols)].itertuples(index=False, name=None)]
    elif column is not None:
        if column not in data:
            raise ValueError(f"Ground-truth column '{column}' is not present in the selected CSV.")
        source = column
        raw_values = data[column].tolist()
    else:
        return {}, 0, None
    for point_number, value in enumerate(raw_values, start=1):
        label = _canonical_label(value)
        if label is not None:
            labels[point_number] = label
    return labels, len(labels), source


def _set_delays(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Attach causal observation and timestamp delays to a baseline output."""
    result = frame.copy()
    result["finalized_at_point"] = result.point_number
    result["prediction_delay_observations"] = 0
    result["prediction_delay_seconds"] = np.nan
    for _, group in result.groupby("segment_id", sort=False, dropna=False):
        ordered = group.sort_values("point_number", kind="stable")
        end_index = len(ordered) - 1 if mode == "offline" else 0
        times = pd.to_datetime(ordered.timestamp, errors="coerce", utc=True)
        for local, (index, row) in enumerate(ordered.iterrows()):
            if not bool(row.matched):
                continue
            arrival = end_index if mode == "offline" else local
            result.at[index, "finalized_at_point"] = int(ordered.iloc[arrival].point_number)
            result.at[index, "prediction_delay_observations"] = max(0, arrival - local)
            if pd.notna(times.iloc[local]) and pd.notna(times.iloc[arrival]):
                seconds = (times.iloc[arrival] - times.iloc[local]).total_seconds()
                if seconds >= 0:
                    result.at[index, "prediction_delay_seconds"] = seconds
    return result


def invalid_transition_count(frame: pd.DataFrame, graph: nx.MultiDiGraph) -> int:
    """Count adjacent matched edge pairs with no directed downstream-to-upstream path."""
    count = 0
    reachability: dict[tuple[Any, Any], bool] = {}
    for _, segment in frame.groupby("segment_id", sort=False, dropna=False):
        segment = segment.sort_values("point_number", kind="stable")
        rows = list(segment.itertuples(index=False))
        for prev, curr in zip(rows, rows[1:]):
            if int(curr.point_number) != int(prev.point_number) + 1:
                continue
            before, after = _edge_tuple(prev), _edge_tuple(curr)
            if before is None or after is None:
                continue
            if before == after:
                continue
            source, target = before[1], after[0]
            key = (source, target)
            if key not in reachability:
                try:
                    reachability[key] = nx.has_path(graph, source, target)
                except (nx.NodeNotFound, nx.NetworkXError):
                    reachability[key] = False
            if not reachability[key]:
                count += 1
    return count


def _metric_row(
    *,
    frame: pd.DataFrame,
    raw: pd.DataFrame,
    graph: nx.MultiDiGraph,
    trajectory_id: str,
    model: str,
    preprocessing_seconds: float,
    decoding_seconds: float,
    ground_truth_column: str | None,
    window_size: int | None = None,
    buffer_size: int | None = None,
) -> dict[str, Any]:
    matched = frame[frame.matched.astype(bool)]
    distances = pd.to_numeric(matched.distance_meters, errors="coerce").dropna()
    labels, label_count, label_source = ground_truth_labels(raw, ground_truth_column)
    exact, compared = 0, 0
    by_point = frame.set_index("point_number")
    for point_number, truth in labels.items():
        compared += 1
        if point_number in by_point.index:
            prediction = _edge_tuple(by_point.loc[point_number])
            if prediction is not None and tuple(map(str, prediction)) == truth:
                exact += 1
    accuracy = (100 * exact / compared) if compared else np.nan
    delays = pd.to_numeric(matched.get("prediction_delay_observations", pd.Series(dtype=float)), errors="coerce").dropna()
    time_delays = pd.to_numeric(matched.get("prediction_delay_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
    runtime = float(preprocessing_seconds + decoding_seconds)
    return {
        "trajectory_id": trajectory_id,
        "model": model,
        "window_size": window_size,
        "buffer_size": buffer_size,
        "input_observations": int(len(frame)),
        "matched_observations": int(len(matched)),
        "unmatched_observations": int(len(frame) - len(matched)),
        "coverage_percent": 100 * len(matched) / max(1, len(frame)),
        "ground_truth_labeled_observations": label_count,
        "ground_truth_compared_observations": compared,
        "ground_truth_accuracy_percent": accuracy,
        "ground_truth_error_percent": (100 - accuracy) if compared else np.nan,
        "ground_truth_column": label_source,
        "mean_gps_to_road_m": float(distances.mean()) if len(distances) else np.nan,
        "median_gps_to_road_m": float(distances.median()) if len(distances) else np.nan,
        "preprocessing_seconds": float(preprocessing_seconds),
        "decoding_seconds": float(decoding_seconds),
        "runtime_seconds": runtime,
        "throughput_observations_per_second": len(frame) / runtime if runtime > 0 else np.nan,
        "nominal_buffer_observations": buffer_size if model == MODEL_SLIDING else (0 if model == MODEL_NEAREST else np.nan),
        "mean_prediction_delay_observations": float(delays.mean()) if len(delays) else np.nan,
        "max_prediction_delay_observations": float(delays.max()) if len(delays) else np.nan,
        "mean_prediction_delay_seconds": float(time_delays.mean()) if len(time_delays) else np.nan,
        "timestamp_delay_count": int(len(time_delays)),
        "invalid_transition_count": invalid_transition_count(frame, graph),
        "invalid_transition_definition": METRIC_DEFINITIONS["invalid_transition_count"],
    }


def _font(size: int, bold: bool = False):
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _save_series_plot(data: pd.DataFrame, metric: str, label: str, title: str, path: Path) -> None:
    width, height = 1200, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 116, 1136, 94, 510
    draw.text((left, 28), title, fill="#242238", font=_font(28, True))
    values = pd.to_numeric(data[metric], errors="coerce").dropna()
    if values.empty:
        draw.text((left, 200), "No valid measurements are available for this plot.", fill="#6e6b7e", font=_font(20))
        image.save(path)
        return
    ymin, ymax = float(values.min()), float(values.max())
    pad = max((ymax - ymin) * .12, abs(ymax) * .05, 0.5)
    ymin, ymax = max(0, ymin - pad), ymax + pad
    draw.line((left, top, left, bottom, right, bottom), fill="#727084", width=2)
    for fraction in (0, .25, .5, .75, 1):
        y = int(bottom - fraction * (bottom - top))
        value = ymin + fraction * (ymax - ymin)
        draw.line((left, y, right, y), fill="#eceaf0", width=1)
        draw.text((14, y - 9), f"{value:.1f}", fill="#6e6b7e", font=_font(15))
    draw.text((left, 612), "Buffer N (future observations)", fill="#514e63", font=_font(17))
    draw.text((12, 72), label, fill="#514e63", font=_font(17))
    configurations = data.dropna(subset=["window_size", "buffer_size", metric]).groupby(["window_size", "buffer_size"], as_index=False)[metric].mean()
    windows = sorted(int(v) for v in configurations.window_size.unique())
    palette = ["#5c4ad1", "#f06463", "#168f8a", "#e3a52e", "#3477c5"]
    min_buffer, max_buffer = int(configurations.buffer_size.min()), int(configurations.buffer_size.max())
    for tick in sorted(configurations.buffer_size.unique()):
        x = int(left + (float(tick) - min_buffer) / max(1, max_buffer - min_buffer) * (right - left))
        draw.text((x - 6, bottom + 10), str(int(tick)), fill="#6e6b7e", font=_font(14))
    for idx, window in enumerate(windows):
        group = configurations[configurations.window_size == window].sort_values("buffer_size")
        pts = []
        for row in group.itertuples(index=False):
            x = int(left + (float(row.buffer_size) - min_buffer) / max(1, max_buffer - min_buffer) * (right - left))
            y = int(bottom - (float(getattr(row, metric)) - ymin) / max(1e-9, ymax - ymin) * (bottom - top))
            pts.append((x, y))
        color = palette[idx % len(palette)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        for x, y in pts:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=1)
        legend_x = left + idx * 170
        draw.line((legend_x, 570, legend_x + 28, 570), fill=color, width=4)
        draw.text((legend_x + 36, 558), f"T = {window}", fill="#514e63", font=_font(15))
    image.save(path)


def save_experiment_plots(results: pd.DataFrame, out_dir: Path) -> list[str]:
    """Save plots only from measured rows; accuracy is omitted without labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sliding = results[results.model == MODEL_SLIDING]
    plots: list[str] = []
    for metric, label, title, filename in (
        ("coverage_percent", "Coverage (%)", "Coverage by buffer and window", "coverage_by_buffer.png"),
        ("mean_prediction_delay_observations", "Mean delay (observations)", "Prediction delay by buffer and window", "delay_by_buffer.png"),
        ("runtime_seconds", "Runtime (seconds)", "End-to-end processing time by buffer and window", "runtime_by_buffer.png"),
    ):
        _save_series_plot(sliding, metric, label, title, out_dir / filename)
        plots.append(filename)
    labeled = results[results.ground_truth_accuracy_percent.notna()]
    if not labeled.empty:
        _save_series_plot(labeled[labeled.model == MODEL_SLIDING], "ground_truth_accuracy_percent", "Exact edge accuracy (%)", "Ground-truth accuracy by buffer and window", out_dir / "accuracy_by_buffer.png")
        plots.append("accuracy_by_buffer.png")
    else:
        (out_dir / "ground_truth_accuracy_unavailable.txt").write_text(
            "No verified directed road-edge labels were supplied. Accuracy/error plots are intentionally omitted. Coverage and GPS-to-road distance are proxy metrics, not accuracy.\n",
            encoding="utf-8",
        )
    return plots


def _new_run_dir(base: Path = EXPERIMENT_ROOT) -> Path:
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    candidate = base / stamp
    suffix = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def run_delay_experiment(
    trajectories: Iterable[tuple[str, pd.DataFrame]],
    *,
    window_sizes: Iterable[int] = (5, 10, 15),
    buffer_sizes: Iterable[int] = (0, 1, 2, 3, 5, 7, 10),
    baselines: Iterable[str] = (MODEL_NEAREST, MODEL_FULL, MODEL_SLIDING),
    ground_truth_column: str | None = None,
    radius_meters: float = DEFAULT_RADIUS_METERS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    gps_sigma_meters: float = GPS_SIGMA_METERS,
    graph_context: Any = None,
    feature_weights: FeatureWeights = DEFAULT_FEATURE_WEIGHTS,
    output_dir: str | Path | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run shared candidate generation once and compare all selected algorithms."""
    selected = set(baselines)
    allowed = {MODEL_NEAREST, MODEL_FULL, MODEL_SLIDING}
    if not selected or not selected <= allowed:
        raise ValueError(f"Select one or more valid baselines: {sorted(allowed)}")
    windows = sorted(set(int(value) for value in window_sizes))
    buffers = sorted(set(int(value) for value in buffer_sizes))
    configurations = [(t, n) for t in windows for n in buffers if 0 <= n < t]
    if MODEL_SLIDING in selected and not configurations:
        raise ValueError("No valid sliding-window settings. Choose T >= 1 and at least one N where 0 <= N < T.")
    trajectories = list(trajectories)
    if not trajectories:
        raise ValueError("Select at least one trajectory for evaluation.")
    feature_weights.validate()
    rows: list[dict[str, Any]] = []
    run_dir = Path(output_dir) if output_dir is not None else _new_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    total_steps = len(trajectories) * (
        1 + int(MODEL_NEAREST in selected) + int(MODEL_FULL in selected)
        + (len(configurations) if MODEL_SLIDING in selected else 0)
    )
    completed_steps = 0

    def report(message: str) -> None:
        if on_progress:
            on_progress(completed_steps, total_steps, message)

    for index, (trajectory_id, raw) in enumerate(trajectories, start=1):
        report(f"Preparing {trajectory_id} ({index}/{len(trajectories)})")
        prepared = generate_trajectory_candidates(
            raw,
            radius_meters=radius_meters,
            max_candidates=max_candidates,
            gps_sigma_meters=gps_sigma_meters,
            graph_context=graph_context,
        )
        preprocessing_seconds = float(prepared["preprocessing_seconds"])
        graph = prepared["graph"]
        completed_steps += 1
        report(f"Prepared {trajectory_id}")
        if MODEL_NEAREST in selected:
            started = time.perf_counter()
            nearest = _set_delays(nearest_candidate_predictions(prepared), "online")
            decoder_seconds = time.perf_counter() - started
            rows.append(_metric_row(frame=nearest, raw=raw, graph=graph, trajectory_id=trajectory_id,
                                    model=MODEL_NEAREST, preprocessing_seconds=preprocessing_seconds,
                                    decoding_seconds=decoder_seconds, ground_truth_column=ground_truth_column))
            completed_steps += 1
            report(f"Nearest-road baseline · {trajectory_id}")
        if MODEL_FULL in selected:
            full_result = decode_network_hmm(prepared)
            full = _set_delays(full_result["observations"], "offline")
            rows.append(_metric_row(frame=full, raw=raw, graph=graph, trajectory_id=trajectory_id,
                                    model=MODEL_FULL, preprocessing_seconds=preprocessing_seconds,
                                    decoding_seconds=float(full_result["decoding_seconds"]), ground_truth_column=ground_truth_column))
            completed_steps += 1
            report(f"Full-sequence HMM · {trajectory_id}")
        if MODEL_SLIDING in selected:
            for t, n in configurations:
                report(f"{trajectory_id}: sliding window T={t}, N={n}")
                result = run_sliding_window_viterbi(
                    prepared, window_size=t, buffer_size=n,
                    gps_sigma_meters=gps_sigma_meters, weights=feature_weights,
                )
                rows.append(_metric_row(frame=result["observations"], raw=raw, graph=graph,
                                        trajectory_id=trajectory_id, model=MODEL_SLIDING,
                                        preprocessing_seconds=preprocessing_seconds,
                                        decoding_seconds=result["decoding_seconds"],
                                        ground_truth_column=ground_truth_column,
                                        window_size=t, buffer_size=n))
                completed_steps += 1
                report(f"Completed {trajectory_id}: T={t}, N={n}")

    results = pd.DataFrame(rows)
    results.to_csv(run_dir / "experiment_results.csv", index=False)
    serializable = results.replace({np.nan: None}).to_dict(orient="records")
    (run_dir / "experiment_results.json").write_text(json.dumps(serializable, indent=2, allow_nan=False, default=str), encoding="utf-8")
    config = {
        "trajectory_ids": [name for name, _ in trajectories],
        "window_sizes": windows,
        "buffer_sizes": buffers,
        "valid_sliding_configurations": configurations,
        "baselines": sorted(selected),
        "ground_truth_column": ground_truth_column,
        "candidate_radius_meters": radius_meters,
        "max_candidates": max_candidates,
        "gps_sigma_meters": gps_sigma_meters,
        "feature_weights": feature_weights.__dict__,
        "feature_weights_status": "Untrained heuristic defaults; not the original paper's learned SVM weights.",
        "metric_definitions": METRIC_DEFINITIONS,
        "runtime_definition": METRIC_DEFINITIONS["runtime_seconds"],
        "window_buffer_definition": "At arrival t, decode at most the trailing T observations and finalize through t-N; flush the remaining tail at segment end or a candidate-free gap.",
    }
    (run_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    plots = save_experiment_plots(results, run_dir)
    return {"results": results, "output_dir": run_dir, "plots": plots, "config": config}
