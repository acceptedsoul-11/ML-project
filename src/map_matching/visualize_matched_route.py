
from pathlib import Path

import folium
import osmnx as ox
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString, box


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    ROOT / "data" / "road_network" / "beijing_drive.graphml"
)

CANDIDATES_FILE = (
    ROOT / "data" / "processed" / "map_matching"
    / "20081023025304_candidates.csv"
)

MATCHED_FILE = (
    ROOT / "data" / "processed" / "map_matching"
    / "20081023025304_matched_route.csv"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed" / "map_matching"
    / "20081023025304_matched_route_map.html"
)

ROAD_BUFFER_METERS = 250


# ============================================================
# HELPERS
# ============================================================

def get_edge_geometry(graph, u, v, data):
    """Return an edge's geometry, including missing geometries."""
    geometry = data.get("geometry")

    if geometry is not None:
        return geometry

    node_u = graph.nodes[u]
    node_v = graph.nodes[v]

    return LineString([
        (node_u["x"], node_u["y"]),
        (node_v["x"], node_v["y"]),
    ])


def draw_road_layer(graph_edges, bounds, transformer, layer):
    """Draw local OSM roads within the requested bounds."""
    count = 0

    for geometry in graph_edges.values():
        if not geometry.intersects(bounds):
            continue

        coordinates = [
            transformer.transform(x, y)[::-1]
            for x, y in geometry.coords
        ]

        folium.PolyLine(
            coordinates,
            color="#cbd2da",
            weight=2,
            opacity=0.8,
            interactive=False,
        ).add_to(layer)

        count += 1

    layer.add_to(layer._parent if False else layer)

    return count


# ============================================================
# MAIN
# ============================================================

def main():
    print("Creating clean offline map...")

    # --------------------------------------------------------
    # 1. Validate and load input files
    # --------------------------------------------------------

    for path in [GRAPH_FILE, CANDIDATES_FILE, MATCHED_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    gps_data = pd.read_csv(CANDIDATES_FILE)
    matched = pd.read_csv(MATCHED_FILE)

    if gps_data.empty:
        raise ValueError("The GPS observations file is empty.")

    if matched.empty:
        raise ValueError("The Viterbi matched-route file is empty.")

    gps_required = {
        "user_id", "trajectory_id", "segment_id",
        "point_number", "gps_latitude", "gps_longitude",
    }

    matched_required = {
        "user_id", "trajectory_id", "segment_id",
        "point_number", "gps_latitude", "gps_longitude",
        "u", "v", "key",
    }

    missing_gps = gps_required - set(gps_data.columns)
    missing_matched = matched_required - set(matched.columns)

    if missing_gps:
        raise ValueError(
            f"GPS data missing columns: {sorted(missing_gps)}"
        )

    if missing_matched:
        raise ValueError(
            f"Matched data missing columns: {sorted(missing_matched)}"
        )

    group_columns = ["user_id", "trajectory_id", "segment_id"]

    # Candidate CSV is used ONLY to recover the original GPS points.
    # Candidate road edges and candidate markers are never drawn.
    gps_data = gps_data.sort_values(
        group_columns + ["point_number"],
        kind="stable",
    ).drop_duplicates(
        group_columns + ["point_number"]
    )

    matched = matched.sort_values(
        group_columns + ["point_number"],
        kind="stable",
    )

    # --------------------------------------------------------
    # 2. Load and project the local road network
    # --------------------------------------------------------

    print("Loading local OSM road network...")

    graph = ox.load_graphml(GRAPH_FILE)
    projected_graph = ox.project_graph(graph)

    to_projected = Transformer.from_crs(
        "EPSG:4326",
        projected_graph.graph["crs"],
        always_xy=True,
    )

    to_geographic = Transformer.from_crs(
        projected_graph.graph["crs"],
        "EPSG:4326",
        always_xy=True,
    )

    edge_geometries = {}

    for u, v, key, data in projected_graph.edges(
        keys=True,
        data=True,
    ):
        edge_geometries[(str(u), str(v), str(key))] = (
            get_edge_geometry(projected_graph, u, v, data)
        )

    # --------------------------------------------------------
    # 3. Calculate the displayed area in projected metres
    # --------------------------------------------------------

    projected_points = [
        to_projected.transform(lon, lat)
        for lon, lat in zip(
            gps_data["gps_longitude"],
            gps_data["gps_latitude"],
        )
    ]

    xs = [point[0] for point in projected_points]
    ys = [point[1] for point in projected_points]

    bounds = box(
        min(xs) - ROAD_BUFFER_METERS,
        min(ys) - ROAD_BUFFER_METERS,
        max(xs) + ROAD_BUFFER_METERS,
        max(ys) + ROAD_BUFFER_METERS,
    )

    center = [
        gps_data["gps_latitude"].mean(),
        gps_data["gps_longitude"].mean(),
    ]

    # No basemap tiles: fully offline.
    route_map = folium.Map(
        location=center,
        zoom_start=14,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # --------------------------------------------------------
    # 4. Background: local OSM roads
    # --------------------------------------------------------

    print("Drawing local roads...")

    roads_layer = folium.FeatureGroup(
        name="Local OSM roads",
        show=True,
    )

    road_count = 0

    for geometry in edge_geometries.values():
        if not geometry.intersects(bounds):
            continue

        coordinates = [
            to_geographic.transform(x, y)[::-1]
            for x, y in geometry.coords
        ]

        folium.PolyLine(
            coordinates,
            color="#cbd2da",
            weight=2,
            opacity=0.8,
            interactive=False,
        ).add_to(roads_layer)

        road_count += 1

    roads_layer.add_to(route_map)

    # --------------------------------------------------------
    # 5. Layer 1: original GPS trace
    # --------------------------------------------------------

    print("Drawing original GPS trace...")

    gps_layer = folium.FeatureGroup(
        name="Original GPS",
        show=True,
    )

    for _, segment in gps_data.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):
        segment = segment.sort_values("point_number")

        coordinates = list(zip(
            segment["gps_latitude"],
            segment["gps_longitude"],
        ))

        if len(coordinates) >= 2:
            folium.PolyLine(
                coordinates,
                color="#ef5350",
                weight=3,
                opacity=0.95,
                tooltip="Original GPS trace",
            ).add_to(gps_layer)

        if not segment.empty:
            first = segment.iloc[0]
            last = segment.iloc[-1]

            folium.CircleMarker(
                [
                    first["gps_latitude"],
                    first["gps_longitude"],
                ],
                radius=5,
                color="#ffffff",
                weight=2,
                fill=True,
                fill_color="#16a34a",
                fill_opacity=1,
                tooltip="Segment start",
            ).add_to(gps_layer)

            folium.CircleMarker(
                [
                    last["gps_latitude"],
                    last["gps_longitude"],
                ],
                radius=5,
                color="#ffffff",
                weight=2,
                fill=True,
                fill_color="#ea580c",
                fill_opacity=1,
                tooltip="Segment end",
            ).add_to(gps_layer)

    gps_layer.add_to(route_map)

    # --------------------------------------------------------
    # 6. Layer 2: Viterbi-matched route
    # --------------------------------------------------------

    print("Drawing Viterbi-matched route...")

    matched_layer = folium.FeatureGroup(
        name="Viterbi-matched route",
        show=True,
    )

    selected_edges = matched[
        ["u", "v", "key"]
    ].drop_duplicates()

    matched_edge_count = 0

    for _, row in selected_edges.iterrows():
        edge_id = (
            str(row["u"]),
            str(row["v"]),
            str(row["key"]),
        )

        geometry = edge_geometries.get(edge_id)

        if geometry is None:
            continue

        coordinates = [
            to_geographic.transform(x, y)[::-1]
            for x, y in geometry.coords
        ]

        folium.PolyLine(
            coordinates,
            color="#2463eb",
            weight=5,
            opacity=0.98,
            tooltip="Viterbi-selected road",
        ).add_to(matched_layer)

        matched_edge_count += 1

    matched_layer.add_to(route_map)

    # --------------------------------------------------------
    # 7. Clean legend
    # --------------------------------------------------------

    legend = """
    <div style="
        position:fixed;
        bottom:24px;
        left:24px;
        z-index:9999;
        background:rgba(255,255,255,0.97);
        padding:14px 18px;
        border-radius:12px;
        box-shadow:0 3px 16px rgba(15,23,42,0.15);
        font-family:Arial,sans-serif;
        color:#253247;
        min-width:210px;
        line-height:1.9;
    ">
        <div style="font-weight:700;font-size:14px;margin-bottom:7px;">
            GeoLife · HMM Map Matching
        </div>
        <div>
            <span style="color:#ef5350;">━━━</span>
            Original GPS
        </div>
        <div>
            <span style="color:#2463eb;">━━━</span>
            Viterbi-matched route
        </div>
        <div>
            <span style="color:#cbd2da;">━━━</span>
            Local OSM roads
        </div>
        <div style="font-size:11px;color:#64748b;margin-top:6px;">
            Offline map · No external tiles
        </div>
    </div>
    """

    route_map.get_root().html.add_child(
        folium.Element(legend)
    )

    # Only the two comparison layers are toggleable.
    folium.LayerControl(
        collapsed=False,
    ).add_to(route_map)

    # Fit all GPS observations without connecting separate segments.
    route_map.fit_bounds([
        [
            gps_data["gps_latitude"].min(),
            gps_data["gps_longitude"].min(),
        ],
        [
            gps_data["gps_latitude"].max(),
            gps_data["gps_longitude"].max(),
        ],
    ], padding=(35, 35))

    # --------------------------------------------------------
    # 8. Save and report
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    route_map.save(str(OUTPUT_FILE))

    print("\nOffline visualization created successfully.")
    print(f"GPS observations: {len(gps_data)}")
    print(f"Local road geometries drawn: {road_count}")
    print(f"Viterbi road edges drawn: {matched_edge_count}")
    print(f"Output: {OUTPUT_FILE}")
    print("Candidate road overlays: DISABLED")
    print("External basemap tiles: DISABLED")


if __name__ == "__main__":
    main()
