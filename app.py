"""RouteLens Streamlit application for GPS road-candidate and trajectory matching."""

from __future__ import annotations

import io
import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer
from shapely.geometry import box
from streamlit_folium import st_folium

from src.map_matching.inference import (
    BETA_METERS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RADIUS_METERS,
    GPS_SIGMA_METERS,
    load_road_network,
    normalize_trajectory,
    predict_single_point,
    predict_trajectory,
    validate_coordinates,
)
from src.evaluation.delay_accuracy import (
    MODEL_FULL,
    MODEL_NEAREST,
    MODEL_SLIDING,
    run_delay_experiment,
)

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "processed" / "user_000" / "20081023025304.csv"
LOG = logging.getLogger(__name__)

st.set_page_config(
    page_title="RouteLens · GPS Map Matching",
    page_icon="⌖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
:root { --ink:#242238; --muted:#6e6b7e; --line:#e7e4ec; --accent:#5c4ad1; --coral:#f06463; --wash:#f7f5fa; --lime:#d9f578; }
html, body, [class*="css"] { font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); }
h1,h2,h3 { letter-spacing:-.035em; }
.block-container { padding-top:1.15rem; padding-bottom:2.5rem; max-width:1500px; }
[data-testid="stSidebar"] { background:#f1eff8; border-right:1px solid var(--line); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { color:#302b55; letter-spacing:-.04em; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:15px; padding:13px 16px; box-shadow:0 3px 12px rgba(36,34,56,.035); }
[data-testid="stMetricLabel"] { color:var(--muted); font-size:.78rem; font-weight:600; }
[data-testid="stMetricValue"] { font-size:1.4rem; font-weight:750; letter-spacing:-.035em; }
div[data-testid="stForm"] { border:1px solid var(--line); border-radius:16px; padding:18px; background:#fff; box-shadow:0 6px 24px rgba(36,34,56,.045); }
div[data-testid="stForm"] [data-testid="stWidgetLabel"] p,
div[data-testid="stForm"] label,
div[data-testid="stForm"] [data-testid="stMarkdownContainer"] p { color:#39364d !important; }
div[data-testid="stForm"] input { color:#242238 !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] p { color:#514e63; }
[data-testid="stRadio"] label p { font-weight:600; }
.hero { background:linear-gradient(112deg,#292644 0%,#433a78 68%,#5548a5 100%); color:#fff; border-radius:20px; padding:20px 26px 21px; margin:0 0 1.15rem; position:relative; overflow:hidden; }
.hero::after { content:""; position:absolute; width:220px; height:220px; border:1px solid rgba(217,245,120,.24); border-radius:50%; right:54px; top:-126px; box-shadow:0 0 0 28px rgba(217,245,120,.045),0 0 0 58px rgba(217,245,120,.03); }
.eyebrow { color:var(--lime); text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:800; margin-bottom:.35rem; }
.hero-title { font-size:1.85rem; line-height:1.12; letter-spacing:-.045em; font-weight:800; color:#fff; }
.hero-subtitle { color:#e3e0f1; font-size:.9rem; margin-top:.42rem; }
.hero-chip { display:inline-block; margin-top:.75rem; background:var(--lime); color:#282544; border-radius:999px; padding:4px 10px; font-size:.68rem; font-weight:750; letter-spacing:.02em; }
.panel-title { font-weight:750; letter-spacing:-.025em; font-size:1.02rem; margin:0 0 .6rem; color:#302b55; }
button[kind="primary"] { background:var(--coral); border-color:var(--coral); border-radius:11px; font-weight:700; }
button[kind="primary"]:hover { background:#dc5154; border-color:#dc5154; color:white; }
[data-testid="stBaseButton-secondary"] { border-radius:10px; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; }
footer { visibility:hidden; }
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the saved Beijing road network…")
def cached_network() -> tuple[Any, Any, Any, Any]:
    return load_road_network()


@st.cache_data(show_spinner=False)
def trajectory_catalog() -> list[str]:
    return sorted(
        str(path.relative_to(ROOT))
        for path in (ROOT / "data" / "processed").glob("user_*/*.csv")
        if path.is_file()
    )


@st.cache_data(show_spinner=False)
def read_trajectory(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(ROOT / path)


def edge_geometry(graph: Any, row: pd.Series) -> Any | None:
    u, v, key = row.get("u"), row.get("v"), row.get("key")
    if pd.isna(u) or pd.isna(v) or pd.isna(key):
        return None
    try:
        u = int(u) if str(u).isdigit() else u
        v = int(v) if str(v).isdigit() else v
        key = int(key) if str(key).isdigit() else key
        data = graph.get_edge_data(u, v, key)
        if not data:
            return None
        geometry = data.get("geometry")
        if geometry is not None:
            return geometry
        from shapely.geometry import LineString
        return LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])])
    except (KeyError, TypeError, ValueError):
        return None


def draw_geometry(layer: folium.FeatureGroup, geometry: Any, transformer: Any, **kwargs: Any) -> None:
    if geometry is None or geometry.is_empty:
        return
    parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
    for part in parts:
        if part.geom_type != "LineString":
            continue
        coords = [transformer.transform(x, y)[::-1] for x, y in part.coords]
        folium.PolyLine(coords, **kwargs).add_to(layer)


def add_map_frame(fmap: folium.Map, graph: Any, edges: Any, to_wgs84: Any,
                  latitudes: list[float], longitudes: list[float], *,
                  predictions: pd.DataFrame | None = None,
                  candidates: pd.DataFrame | None = None,
                  point: tuple[float, float] | None = None) -> None:
    """Add locally saved roads and the actual prediction overlays."""
    to_projected = Transformer.from_crs(
        "EPSG:4326", graph.graph["crs"], always_xy=True
    )
    projected = [to_projected.transform(lon, lat) for lat, lon in zip(latitudes, longitudes)]
    pad = 350.0 if len(projected) < 2 else 220.0
    bounds = box(min(x for x, _ in projected) - pad, min(y for _, y in projected) - pad,
                 max(x for x, _ in projected) + pad, max(y for _, y in projected) + pad)
    road_layer = folium.FeatureGroup(name="Saved OSM roads", show=True)
    try:
        positions = edges.sindex.query(bounds, predicate="intersects")
    except (AttributeError, ValueError):
        positions = []
    for position in list(positions)[:2500]:
        draw_geometry(road_layer, edges.geometry.iloc[int(position)], to_wgs84,
                      color="#bdc8ce", weight=1.4, opacity=.68, interactive=False)
    road_layer.add_to(fmap)

    if point is not None and candidates is not None and not candidates.empty:
        candidate_layer = folium.FeatureGroup(name="Nearby road candidates", show=True)
        for rank, (_, candidate) in enumerate(candidates.iterrows(), start=1):
            geometry = edge_geometry(graph, candidate)
            label = candidate.get("road_name") or "Unnamed road"
            road_type = candidate.get("road_type") or "Unknown type"
            distance = float(candidate["distance_meters"])
            score = float(candidate["emission_probability"])
            draw_geometry(candidate_layer, geometry, to_wgs84,
                          color="#0b7180" if rank == 1 else "#e69541",
                          weight=6 if rank == 1 else 4, opacity=.95,
                          tooltip=f"{rank}. {label} · {distance:.1f} m · {score:.0%}",
                          popup=f"{label}<br>{road_type}<br>{distance:.1f} m")
        candidate_layer.add_to(fmap)
        folium.CircleMarker(point, radius=7, color="#fff", weight=2,
                            fill=True, fill_color="#c74440", fill_opacity=1,
                            tooltip="GPS observation").add_to(fmap)
        return

    gps_layer = folium.FeatureGroup(name="Original GPS trace", show=True)
    gps_points_layer = folium.FeatureGroup(name="Actual GPS observations", show=True)
    matched_layer = folium.FeatureGroup(name="Viterbi matched roads", show=True)
    start_end_layer = folium.FeatureGroup(name="Trajectory endpoints", show=True)
    if predictions is not None and not predictions.empty:
        for segment_id, segment in predictions.groupby("segment_id", sort=False):
            segment = segment.sort_values("point_number", kind="stable")
            coords = list(zip(segment.latitude.astype(float), segment.longitude.astype(float)))
            if len(coords) > 1:
                folium.PolyLine(coords, color="#c74440", weight=3, opacity=.85,
                                tooltip=f"Original GPS · segment {segment_id}").add_to(gps_layer)
            for row in segment.itertuples(index=False):
                is_matched = bool(row.matched)
                if is_matched:
                    road = row.road_name if pd.notna(row.road_name) else "Unnamed road"
                    distance = float(row.distance_meters) if pd.notna(row.distance_meters) else None
                    detail = f"Matched to {road}" + (f" · {distance:.1f} m away" if distance is not None else "")
                    fill_color = "#5c4ad1"
                else:
                    detail = "No road candidate within search radius"
                    fill_color = "#f06463"
                folium.CircleMarker(
                    [float(row.latitude), float(row.longitude)], radius=3.5,
                    color="white", weight=1, fill=True, fill_color=fill_color,
                    fill_opacity=.95,
                    tooltip=f"GPS observation #{int(row.point_number)} · {detail}",
                ).add_to(gps_points_layer)
            for sequence_block, block in segment[segment.matched].groupby("sequence_block", sort=False):
                selected_edges = block.drop_duplicates(["u", "v", "key"])
                for _, match in selected_edges.iterrows():
                    road = match.get("road_name") or "Unnamed road"
                    draw_geometry(matched_layer, edge_geometry(graph, match), to_wgs84,
                                  color="#1769aa", weight=5, opacity=.95,
                                  tooltip=f"{road} · {float(match.distance_meters):.1f} m")
            if len(segment):
                first, last = segment.iloc[0], segment.iloc[-1]
                folium.CircleMarker([first.latitude, first.longitude], radius=5, color="white", weight=2,
                                    fill=True, fill_color="#248451", fill_opacity=1,
                                    tooltip="Trajectory start").add_to(start_end_layer)
                folium.CircleMarker([last.latitude, last.longitude], radius=5, color="white", weight=2,
                                    fill=True, fill_color="#df8736", fill_opacity=1,
                                    tooltip="Trajectory end").add_to(start_end_layer)
    gps_layer.add_to(fmap)
    matched_layer.add_to(fmap)
    gps_points_layer.add_to(fmap)
    start_end_layer.add_to(fmap)


def make_single_map(lat: float, lon: float, candidates: pd.DataFrame,
                    graph: Any, edges: Any, to_wgs84: Any) -> folium.Map:
    fmap = folium.Map(location=[lat, lon], zoom_start=16, tiles="OpenStreetMap",
                      control_scale=True, prefer_canvas=True)
    add_map_frame(fmap, graph, edges, to_wgs84, [lat], [lon], candidates=candidates,
                  point=(lat, lon))
    fmap.fit_bounds([[lat - .004, lon - .004], [lat + .004, lon + .004]])
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def make_trajectory_map(result: dict[str, Any]) -> folium.Map:
    frame = result["observations"]
    latitudes, longitudes = frame.latitude.astype(float).tolist(), frame.longitude.astype(float).tolist()
    center = [float(np.mean(latitudes)), float(np.mean(longitudes))]
    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap",
                      control_scale=True, prefer_canvas=True)
    add_map_frame(fmap, result["graph"], result["edges"], result["to_wgs84"],
                  latitudes, longitudes, predictions=frame)
    fmap.fit_bounds([[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]], padding=(25, 25))
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def clean_download(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def frame_signature(source_identity: str, frame: pd.DataFrame, radius: int,
                   max_candidates: int) -> str:
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    payload = f"{source_identity}|{radius}|{max_candidates}|".encode() + values
    return hashlib.sha256(payload).hexdigest()


def metric_row(frame: pd.DataFrame, runtime: float) -> None:
    matched = frame[frame.matched]
    dists = matched.distance_meters.dropna()
    cols = st.columns(5)
    values = [
        ("Observations", f"{len(frame):,}"),
        ("Matched", f"{len(matched):,}"),
        ("Coverage", f"{100 * len(matched) / max(1, len(frame)):.1f}%"),
        ("Mean distance", f"{dists.mean():.1f} m" if len(dists) else "—"),
        ("Runtime", f"{runtime:.2f} s"),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, value)


def show_trajectory_result(result: dict[str, Any]) -> None:
    frame = result["observations"]
    matched = frame[frame.matched]
    metric_row(frame, result["runtime_seconds"])
    if matched.empty:
        st.warning("No observations matched roads in the saved graph and selected search radius.")
    else:
        distances = matched.distance_meters.dropna()
        st.caption(
            f"Median GPS to selected road: {distances.median():.1f} m · "
            f"Unmatched: {len(frame) - len(matched):,} · "
            "GPS to road distance describes geometric fit, not ground-truth accuracy."
        )
    st_folium(make_trajectory_map(result), height=590, use_container_width=True,
              returned_objects=[])
    st.caption("Map key: purple dots are matched GPS observations; coral dots are unmatched. Hover a dot for point details, or toggle layers from the map control.")
    with st.expander("Per-point predictions", expanded=True):
        display = frame.copy()
        display["status"] = np.where(display.matched, "Matched", "Unmatched")
        st.dataframe(display, use_container_width=True, hide_index=True, height=320)
    st.download_button("Download matched-route CSV", clean_download(frame),
                       file_name="routelens_matched_route.csv", mime="text/csv")
    summaries = frame.groupby("segment_id", sort=False).agg(
        observations=("point_number", "size"), matched=("matched", "sum")
    )
    if len(summaries) > 1:
        st.subheader("Segment summary")
        summaries["coverage_percent"] = 100 * summaries.matched / summaries.observations
        st.dataframe(summaries.reset_index(), use_container_width=True, hide_index=True)


def parse_manual_points(raw: str) -> pd.DataFrame:
    rows = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) not in (2, 3):
            raise ValueError(f"Line {line_number}: use latitude, longitude or timestamp, latitude, longitude.")
        if len(fields) == 2:
            lat, lon = fields
            timestamp = ""
        else:
            timestamp, lat, lon = fields
        validate_coordinates(lat, lon)
        rows.append({"timestamp": timestamp, "latitude": float(lat), "longitude": float(lon)})
    if not rows:
        raise ValueError("Enter at least one GPS point.")
    return pd.DataFrame(rows)


def page_header() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">ROUTELENS · BEIJING ROAD GRAPH</div>'
        '<div class="hero-title">Your GPS dots, meet the streets.</div>'
        '<div class="hero-subtitle">Match one location or a full GeoLife trail to roads on the saved map.</div>'
        '<span class="hero-chip">HMM + VITERBI · LOCAL DEMO</span></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    with st.sidebar:
        st.markdown("## RouteLens")
        st.caption("Your trip, mapped to real streets")
        page = st.radio("Workspace", ["Map Matching", "Trajectory Explorer", "Evaluation", "Buffer Experiments"], label_visibility="collapsed")
        st.divider()
        st.markdown("**Model settings**")
        radius = st.slider("Candidate search radius", 25, 250, int(DEFAULT_RADIUS_METERS), 25, format="%d m")
        max_candidates = st.slider("Candidates per observation", 1, 10, DEFAULT_MAX_CANDIDATES)
        st.caption(f"Emission σ {GPS_SIGMA_METERS:.0f} m · Transition β {BETA_METERS:.0f} m")
        st.divider()
        st.caption("Road graph · saved Beijing OSM driving network")
        if st.button("Check graph status", use_container_width=True):
            try:
                graph, edges, _, _ = cached_network()
                st.success(f"Ready · {len(graph):,} nodes · {len(edges):,} directed roads")
            except Exception as exc:
                st.error(str(exc))

    page_header()
    if page == "Map Matching":
        render_map_matching(radius, max_candidates)
    elif page == "Trajectory Explorer":
        render_explorer(radius, max_candidates)
    elif page == "Evaluation":
        render_evaluation()
    else:
        render_buffer_experiments(radius, max_candidates)


def render_map_matching(radius: int, max_candidates: int) -> None:
    mode = st.radio("Prediction mode", ["Single GPS point", "Trajectory"], horizontal=True)
    if mode == "Single GPS point":
        left, right = st.columns([.8, 1.7], gap="large")
        with left:
            st.markdown('<p class="panel-title">Drop a pin</p>', unsafe_allow_html=True)
            st.caption("One point finds nearby road picks. A full trail runs HMM/Viterbi.")
            with st.form("single_point_form"):
                default_lat, default_lon = 39.984702, 116.318417
                if SAMPLE.exists():
                    try:
                        sample = pd.read_csv(SAMPLE, nrows=1).iloc[0]
                        default_lat, default_lon = float(sample.latitude), float(sample.longitude)
                    except (OSError, ValueError, KeyError, IndexError):
                        pass
                st.caption("We start you at the first point in the saved GeoLife sample.")
                lat = st.number_input("Latitude", value=default_lat, format="%.7f")
                lon = st.number_input("Longitude", value=default_lon, format="%.7f")
                timestamp = st.text_input("Timestamp · optional", placeholder="2026-10-12 08:30:00")
                sigma = st.number_input("GPS accuracy σ · metres", min_value=5.0, max_value=100.0, value=GPS_SIGMA_METERS, step=5.0)
                submitted = st.form_submit_button("Find nearby roads", type="primary", use_container_width=True)
            single_signature = (float(lat), float(lon), timestamp, int(radius),
                                int(max_candidates), float(sigma))
            if submitted:
                st.session_state.pop("single_result", None)
                try:
                    lat, lon = validate_coordinates(lat, lon)
                    network = cached_network()
                    candidates = predict_single_point(lat, lon, radius_meters=radius,
                                                      max_candidates=max_candidates,
                                                      gps_sigma_meters=sigma,
                                                      graph_context=network)
                    st.session_state.single_result = {"lat": lat, "lon": lon, "timestamp": timestamp,
                                                      "candidates": candidates, "network": network,
                                                      "signature": single_signature}
                except Exception as exc:
                    LOG.exception("Single point prediction failed")
                    st.error(str(exc))
            saved = st.session_state.get("single_result")
            if saved and saved.get("signature") == single_signature:
                candidates = saved["candidates"]
                if candidates.empty:
                    st.warning(f"No road candidates found within {radius} m of this point.")
                else:
                    best = candidates.iloc[0]
                    st.metric("Closest road", best.road_name or "Unnamed road")
                    st.write(f"**Type:** {best.road_type or 'Unknown'}  ·  **Distance:** {best.distance_meters:.1f} m")
                    st.caption(f"{len(candidates)} road candidates · score normalized across nearby candidates")
                    st.dataframe(candidates[["candidate_rank", "road_name", "road_type", "distance_meters", "emission_probability"]],
                                 use_container_width=True, hide_index=True)
        with right:
            saved = st.session_state.get("single_result")
            if saved and saved.get("signature") == single_signature:
                graph, edges, _, to_wgs84 = saved["network"]
                st_folium(make_single_map(saved["lat"], saved["lon"], saved["candidates"], graph, edges, to_wgs84),
                          height=620, use_container_width=True, returned_objects=[])
            else:
                st.info("Enter a location and run a search to see candidate roads on the map.")
        return

    render_trajectory_inputs(radius, max_candidates)


def render_trajectory_inputs(radius: int, max_candidates: int) -> None:
    catalog = trajectory_catalog()
    source = st.radio("Trajectory source", ["Saved GeoLife", "Upload CSV", "Enter points"], horizontal=True)
    trajectory: pd.DataFrame | None = None
    column_mapping: dict[str, str] = {}
    source_identity = source
    if source == "Saved GeoLife":
        if not catalog:
            st.error("No processed trajectories were found under data/processed/user_*.")
            return
        default = "data/processed/user_000/20081023025304.csv"
        users = sorted({Path(item).parent.name for item in catalog})
        default_user = Path(default).parent.name
        selected_user = st.selectbox("GeoLife user", users,
                                     index=users.index(default_user) if default_user in users else 0)
        available = [item for item in catalog if Path(item).parent.name == selected_user]
        selected = st.selectbox("Processed trajectory", available,
                                index=available.index(default) if default in available else 0)
        source_identity = selected
        path = ROOT / selected
        trajectory = read_trajectory(selected, path.stat().st_mtime)
        st.caption(f"{len(trajectory):,} GPS observations · {selected}")
    elif source == "Upload CSV":
        upload = st.file_uploader("Timestamped GPS CSV", type=["csv"])
        if upload is None:
            st.info("Choose a CSV with latitude and longitude columns. Timestamp and segment columns are optional.")
            return
        try:
            uploaded_bytes = upload.getvalue()
            trajectory = pd.read_csv(io.BytesIO(uploaded_bytes))
            source_identity = upload.name + hashlib.sha256(uploaded_bytes).hexdigest()
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            return
        with st.expander("CSV columns and preview"):
            st.write(", ".join(map(str, trajectory.columns)))
            st.dataframe(trajectory.head(8), use_container_width=True, hide_index=True)
        choices = list(trajectory.columns)
        lowered = {str(column).strip().lower(): column for column in choices}
        lat_default = next((lowered[name] for name in ("latitude", "lat", "gps_latitude", "gps_lat") if name in lowered), choices[0])
        lon_default = next((lowered[name] for name in ("longitude", "lon", "lng", "gps_longitude", "gps_lon") if name in lowered), choices[min(1, len(choices) - 1)])
        time_default = next((lowered[name] for name in ("timestamp", "time", "datetime", "date_time") if name in lowered), None)
        seg_default = next((lowered[name] for name in ("segment_id", "segment", "trip_segment") if name in lowered), None)
        m1, m2, m3, m4 = st.columns(4)
        lat_col = m1.selectbox("Latitude column", choices, index=choices.index(lat_default))
        lon_col = m2.selectbox("Longitude column", choices, index=choices.index(lon_default))
        time_col = m3.selectbox("Timestamp · optional", ["(none)"] + choices,
                                index=0 if time_default is None else choices.index(time_default) + 1)
        seg_col = m4.selectbox("Segment · optional", ["(none)"] + choices,
                               index=0 if seg_default is None else choices.index(seg_default) + 1)
        column_mapping = {"latitude_column": lat_col, "longitude_column": lon_col,
                          "timestamp_column": None if time_col == "(none)" else time_col,
                          "segment_column": None if seg_col == "(none)" else seg_col}
    else:
        raw = st.text_area("GPS points", height=140, placeholder="39.984702, 116.318417\n2008-10-23 02:53:09, 39.984710, 116.318450")
        source_identity = raw
        if raw.strip():
            try:
                trajectory = parse_manual_points(raw)
            except ValueError as exc:
                st.error(str(exc))
                return
        if trajectory is None:
            st.info("Enter one or more points to continue.")
            return

    if trajectory is not None:
        try:
            normalized = normalize_trajectory(trajectory, **column_mapping)
        except ValueError as exc:
            st.error(f"Trajectory validation failed: {exc}")
            return
        signature = frame_signature(source_identity, normalized, radius, max_candidates)
        if st.session_state.get("trajectory_result_signature") != signature:
            st.session_state.pop("trajectory_result", None)
        if st.button("Run HMM map matching", type="primary"):
            st.session_state.pop("trajectory_result", None)
            try:
                with st.spinner(f"Matching {len(normalized):,} GPS observations…"):
                    result = predict_trajectory(trajectory, radius_meters=radius,
                                                max_candidates=max_candidates,
                                                graph_context=cached_network(),
                                                **column_mapping)
                st.session_state.trajectory_result = result
                st.session_state.trajectory_result_signature = signature
                st.session_state.evaluation_result = result
            except Exception as exc:
                LOG.exception("Trajectory prediction failed")
                st.error(f"Map matching failed: {exc}")
        result = st.session_state.get("trajectory_result")
        if result:
            st.divider()
            show_trajectory_result(result)


def render_explorer(radius: int, max_candidates: int) -> None:
    st.subheader("Trajectory Explorer")
    st.caption("Load a processed GeoLife file, upload a CSV, or paste coordinates. Results are computed from the selected observations.")
    render_trajectory_inputs(
        radius,
        max_candidates,
    )


def render_evaluation() -> None:
    st.subheader("Evaluation")
    st.markdown("### Paper reference")
    paper_left, paper_right = st.columns([1, 2], gap="large")
    with paper_left:
        st.metric("Reported classification error", "0.33%–4.6%")
        st.caption("1,227 labeled observations from San Francisco and Palo Alto test drives.")
    with paper_right:
        st.markdown(
            "The supplied paper reports a dense, real-time HMM with a sliding window and buffer. "
            "Its emissions combine GPS-to-road distance, heading alignment, and a speed-limit constraint. "
            "Transitions use backtrack prevention and midpoint-distance discrepancy. The authors learn "
            "feature weights with L1-regularized SVMs (C = 1)."
        )
        st.caption(
            "Reference: Y. Ahres, A. Jambulapati, L. Janssen, and J. Kangaspunta, "
            "“Real-Time Dense Map Matching with Naive Hidden Markov Models: Delay versus Accuracy,” "
            "CS229 final project, Stanford University."
        )
    with st.expander("How this GeoLife implementation relates to the paper"):
        st.markdown(
            "**Shared:** HMM/Viterbi sequence inference, OSM road candidates, and a Gaussian "
            "GPS-to-road emission.\n\n"
            "**Current implementation:** full-trajectory Viterbi, Gaussian distance emissions, and "
            "directed shortest-path distance transitions on the saved Beijing graph.\n\n"
            "**Not reproduced:** sliding-window delay/buffer, heading and speed-limit emissions, "
            "backtrack and midpoint transition features, and the learned SVM weights. The supplied "
            "paper does not report the learned weight values. GeoLife Beijing trajectories also lack "
            "the paper’s labeled road ground truth, so its classification error cannot be claimed here."
        )
    st.divider()
    st.markdown("### Current GeoLife run")
    result = st.session_state.get("evaluation_result") or st.session_state.get("trajectory_result")
    if not result:
        st.info("Run trajectory map matching to populate these metrics from a real prediction.")
        return
    frame = result["observations"]
    matched = frame[frame.matched]
    metric_row(frame, result["runtime_seconds"])
    distances = matched.distance_meters.dropna()
    c1, c2, c3 = st.columns(3)
    c1.metric("Unmatched observations", f"{len(frame) - len(matched):,}")
    c2.metric("Median GPS to road", f"{distances.median():.1f} m" if len(distances) else "—")
    c3.metric("Ground-truth accuracy", "Not evaluated")
    st.caption("Coverage counts observations with a selected road candidate. GPS-to-road distance is geometric fit; it is not ground-truth road accuracy.")
    if len(distances):
        st.subheader("GPS to selected-road distance")
        counts, edges = np.histogram(
            distances.to_numpy(dtype=float),
            bins=min(12, max(3, int(math.sqrt(len(distances))))),
        )
        labels = [f"{edges[i]:.0f}–{edges[i + 1]:.0f} m" for i in range(len(counts))]
        chart = pd.DataFrame({"Observations": counts}, index=labels)
        st.bar_chart(chart)
    summary = frame.groupby("segment_id", sort=False).agg(
        observations=("point_number", "size"), matched=("matched", "sum"),
        mean_road_distance_m=("distance_meters", "mean"),
    ).reset_index()
    summary["coverage_percent"] = 100 * summary.matched / summary.observations
    st.subheader("Per-segment summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_buffer_experiments(radius: int, max_candidates: int) -> None:
    """Interactive, reproducible delay/coverage sweep over processed GPS data."""
    st.subheader("Buffer experiments")
    st.caption("Compare nearest-road, full-trajectory HMM, and delayed sliding-window predictions on processed trajectories.")
    files = trajectory_catalog()
    if not files:
        st.warning("No processed GeoLife CSV files are available in data/processed.")
        return
    with st.form("buffer_experiment_form"):
        users = sorted({Path(path).parts[2] for path in files})
        user = st.selectbox("GeoLife user", users)
        user_files = [path for path in files if Path(path).parts[2] == user]
        selected_paths = st.multiselect("Trajectories", user_files, default=user_files[:1])
        c1, c2 = st.columns(2)
        with c1:
            windows = st.multiselect("Window sizes T", [5, 10, 15, 20], default=[5, 10, 15])
        with c2:
            buffers = st.multiselect("Future buffer sizes N", [0, 1, 2, 3, 5, 7, 10], default=[0, 1, 2, 3, 5])
        models = st.multiselect(
            "Methods", [MODEL_NEAREST, MODEL_FULL, MODEL_SLIDING],
            default=[MODEL_NEAREST, MODEL_FULL, MODEL_SLIDING],
            format_func={MODEL_NEAREST: "Nearest road", MODEL_FULL: "Full-sequence HMM", MODEL_SLIDING: "Sliding-window HMM"}.get,
        )
        gt_options = ["(No ground-truth labels)"]
        if selected_paths:
            preview = read_trajectory(selected_paths[0], (ROOT / selected_paths[0]).stat().st_mtime)
            gt_options += [column for column in preview.columns if any(token in column.lower() for token in ("ground_truth", "true_edge", "reference_edge"))]
        ground_truth = st.selectbox("Optional directed-edge label column", gt_options)
        run = st.form_submit_button("Run experiment", type="primary", use_container_width=True)

    experiment_signature = hashlib.sha256(repr((
        tuple(selected_paths), tuple(windows), tuple(buffers), tuple(models),
        ground_truth, int(radius), int(max_candidates),
    )).encode("utf-8")).hexdigest()
    if run:
        if not selected_paths or not models:
            st.error("Select at least one trajectory and one method.")
            return
        if MODEL_SLIDING in models and not any(0 <= n < t for t in windows for n in buffers):
            st.error("Each sliding-window configuration needs 0 ≤ N < T. Adjust the selected window and buffer sizes.")
            return
        inputs = [
            (path, read_trajectory(path, (ROOT / path).stat().st_mtime))
            for path in selected_paths
        ]
        progress = st.progress(0, text="Starting experiment…")
        status = st.empty()

        def update_progress(done: int, total: int, message: str) -> None:
            progress.progress(min(1.0, done / max(1, total)), text=message)
            status.caption(message)

        try:
            outcome = run_delay_experiment(
                inputs, window_sizes=windows or [5], buffer_sizes=buffers or [0],
                baselines=models, ground_truth_column=None if ground_truth == gt_options[0] else ground_truth,
                radius_meters=float(radius), max_candidates=int(max_candidates),
                graph_context=cached_network(), on_progress=update_progress,
            )
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
            return
        progress.progress(1.0, text="Experiment complete")
        st.session_state["buffer_experiment_result"] = outcome
        st.session_state["buffer_experiment_signature"] = experiment_signature

    outcome = st.session_state.get("buffer_experiment_result")
    if not outcome:
        st.info("Choose trajectories and run a sweep. Results and plots will be saved under results/experiments/.")
        return
    if st.session_state.get("buffer_experiment_signature") != experiment_signature:
        st.info("Experiment settings changed. Run the sweep again to refresh these results.")
        return
    results = outcome["results"]
    st.success(f"Saved reproducible results to `{outcome['output_dir']}`")
    st.download_button(
        "Download results CSV", results.to_csv(index=False).encode("utf-8"),
        file_name="experiment_results.csv", mime="text/csv",
    )
    st.dataframe(results, use_container_width=True, hide_index=True)
    for plot in outcome["plots"]:
        st.image(str(outcome["output_dir"] / plot), caption=plot.replace("_", " ").removesuffix(".png").title(), use_container_width=True)
    if results.ground_truth_accuracy_percent.isna().all():
        st.info("No verified directed-edge labels were supplied. Accuracy/error is left blank; coverage and GPS-to-road distance are fit proxies, not accuracy.")
    with st.expander("Metric definitions and method notes"):
        st.markdown(
            "**Coverage** is the fraction of points assigned a candidate. **GPS-to-road distance** measures geometric fit. "
            "**Prediction delay** is the number of future observations consumed before commitment. Offline baselines are "
            "reported as final at trajectory end. **Invalid transitions** count adjacent chosen roads without a directed "
            "downstream-to-upstream path; the paper-informed naive HMM does not force shortest-path topology."
        )
        st.caption("Feature weights are explicit untrained heuristic defaults, not learned SVM weights from the paper. The CSV, config JSON, and measured-run plots are saved with the experiment.")


if __name__ == "__main__":
    main()
