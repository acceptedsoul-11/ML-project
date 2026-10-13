"""Regression checks for GPS observation markers on the trajectory map."""

import unittest

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from app import make_trajectory_map


class TrajectoryMapTests(unittest.TestCase):
    def test_map_shows_each_observation_with_matched_or_unmatched_detail(self):
        to_projected = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        x, y = to_projected.transform(116.3, 39.9)
        line = LineString([(x, y), (x + 30, y)])
        graph = nx.MultiDiGraph(crs="EPSG:3857")
        graph.add_node(1, x=x, y=y)
        graph.add_node(2, x=x + 30, y=y)
        graph.add_edge(1, 2, key=0, geometry=line, length=30)
        edges = gpd.GeoDataFrame(
            [{"u": 1, "v": 2, "key": 0, "geometry": line}],
            geometry="geometry", crs="EPSG:3857",
        )
        coordinates = [to_wgs84.transform(x + offset, y) for offset in (4, 14, 25)]
        frame = pd.DataFrame([
            {
                "point_number": index + 1, "latitude": latitude, "longitude": longitude,
                "segment_id": "trip", "matched": index < 2,
                "sequence_block": 1 if index < 2 else pd.NA,
                "u": 1 if index < 2 else pd.NA,
                "v": 2 if index < 2 else pd.NA,
                "key": 0 if index < 2 else pd.NA,
                "road_name": "Test Street" if index < 2 else pd.NA,
                "distance_meters": 2.0 if index < 2 else float("nan"),
            }
            for index, (longitude, latitude) in enumerate(coordinates)
        ])

        rendered = make_trajectory_map({
            "observations": frame, "graph": graph, "edges": edges, "to_wgs84": to_wgs84,
        }).get_root().render()

        self.assertIn("Actual GPS observations", rendered)
        for point_number in (1, 2, 3):
            self.assertIn(f"GPS observation #{point_number}", rendered)
        self.assertIn("Matched to Test Street", rendered)
        self.assertIn("No road candidate within search radius", rendered)
        self.assertIn("#5c4ad1", rendered)
        self.assertIn("#f06463", rendered)


if __name__ == "__main__":
    unittest.main()
