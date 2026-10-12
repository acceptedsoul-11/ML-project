from pathlib import Path

import folium
import geopandas as gpd
from shapely.geometry import box

from candidate_generation import (
    load_road_network,
    create_coordinate_transformer,
    find_candidates,
    find_test_gps_point,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "road_network"

OUTPUT_FILE = (
    OUTPUT_DIR / "candidate_visualization.html"
)


# ============================================================
# Configuration
# ============================================================

SEARCH_RADIUS_METERS = 100

# Size of the local road-network area shown around GPS point
DISPLAY_RADIUS_DEGREES = 0.003


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("GPS → ROAD CANDIDATE VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load road network
    # --------------------------------------------------------

    G, edges = load_road_network()

    # --------------------------------------------------------
    # Coordinate transformer
    # --------------------------------------------------------

    transformer = create_coordinate_transformer(
        G.graph["crs"]
    )

    # --------------------------------------------------------
    # Get test GPS point
    # --------------------------------------------------------

    gps = find_test_gps_point()

    if gps is None:
        raise RuntimeError(
            "Could not find a test GPS point."
        )

    latitude = gps["latitude"]
    longitude = gps["longitude"]

    print("\nGPS point:")
    print(f"  Latitude : {latitude}")
    print(f"  Longitude: {longitude}")

    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidates = find_candidates(
        latitude=latitude,
        longitude=longitude,
        edges=edges,
        transformer=transformer,
        radius_meters=SEARCH_RADIUS_METERS,
    )

    if not candidates:
        raise RuntimeError(
            "No candidate roads found."
        )

    print(
        f"\nCandidates found: {len(candidates)}"
    )

    # --------------------------------------------------------
    # Convert complete road network to WGS84
    # --------------------------------------------------------

    edges_wgs84 = edges.to_crs("EPSG:4326")

    # --------------------------------------------------------
    # Select nearby roads for visualization
    # --------------------------------------------------------

    local_bbox = box(
        longitude - DISPLAY_RADIUS_DEGREES,
        latitude - DISPLAY_RADIUS_DEGREES,
        longitude + DISPLAY_RADIUS_DEGREES,
        latitude + DISPLAY_RADIUS_DEGREES,
    )

    local_edges = edges_wgs84[
        edges_wgs84.geometry.intersects(local_bbox)
    ].copy()

    print(
        f"Nearby road segments displayed: "
        f"{len(local_edges):,}"
    )

    # --------------------------------------------------------
    # Create map WITHOUT external tiles
    # --------------------------------------------------------

    fmap = folium.Map(
        location=[
            latitude,
            longitude,
        ],
        zoom_start=16,
        tiles=None,
    )

    # --------------------------------------------------------
    # Draw local OSM roads
    # --------------------------------------------------------

    for _, row in local_edges.iterrows():

        geometry = row.geometry

        if geometry is None:
            continue

        if geometry.geom_type == "LineString":

            coordinates = [
                [lat, lon]
                for lon, lat in geometry.coords
            ]

            folium.PolyLine(
                locations=coordinates,
                color="gray",
                weight=2,
                opacity=0.45,
            ).add_to(fmap)

    # --------------------------------------------------------
    # GPS marker
    # --------------------------------------------------------

    folium.Marker(
        location=[
            latitude,
            longitude,
        ],
        popup=(
            "<b>GeoLife GPS Point</b><br>"
            f"Latitude: {latitude}<br>"
            f"Longitude: {longitude}<br>"
            f"Time: {gps['timestamp']}"
        ),
        tooltip="GPS observation",
        icon=folium.Icon(
            color="red",
            icon="info-sign",
        ),
    ).add_to(fmap)

    # --------------------------------------------------------
    # 100-meter candidate search radius
    # --------------------------------------------------------

    folium.Circle(
        location=[
            latitude,
            longitude,
        ],
        radius=SEARCH_RADIUS_METERS,
        color="blue",
        fill=False,
        weight=2,
        tooltip="100 m candidate search radius",
    ).add_to(fmap)

    # --------------------------------------------------------
    # Candidate roads
    # --------------------------------------------------------

    for i, candidate in enumerate(
        candidates,
        start=1
    ):

        edge_index = (
            candidate["u"],
            candidate["v"],
            candidate["key"],
        )

        edge = edges_wgs84.loc[edge_index]

        geometry = edge.geometry

        if geometry is None:
            continue

        if geometry.geom_type != "LineString":
            continue

        coordinates = [
            [lat, lon]
            for lon, lat in geometry.coords
        ]

        distance = candidate[
            "distance_meters"
        ]

        road_name = candidate[
            "road_name"
        ]

        road_type = candidate[
            "road_type"
        ]

        # First candidate gets a thicker line.
        weight = 7 if i == 1 else 5

        # Candidate 1 is highlighted.
        color = "red" if i == 1 else "blue"

        popup_text = (
            f"<b>Candidate {i}</b><br>"
            f"Distance: {distance:.2f} m<br>"
            f"Road type: {road_type}<br>"
            f"Road name: {road_name}"
        )

        folium.PolyLine(
            locations=coordinates,
            color=color,
            weight=weight,
            opacity=1.0,
            popup=popup_text,
            tooltip=(
                f"Candidate {i} — "
                f"{distance:.2f} m"
            ),
        ).add_to(fmap)

    # --------------------------------------------------------
    # Fit map to local area
    # --------------------------------------------------------

    fmap.fit_bounds(
        [
            [
                latitude - DISPLAY_RADIUS_DEGREES,
                longitude - DISPLAY_RADIUS_DEGREES,
            ],
            [
                latitude + DISPLAY_RADIUS_DEGREES,
                longitude + DISPLAY_RADIUS_DEGREES,
            ],
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fmap.save(OUTPUT_FILE)

    print("\nVisualization saved to:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()