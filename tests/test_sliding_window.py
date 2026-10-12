"""Focused tests for delayed map matching and experiment metrics."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import LineString

from src.evaluation.delay_accuracy import ground_truth_labels, run_delay_experiment
from src.map_matching.sliding_window import (
    nearest_candidate_predictions,
    run_sliding_window_viterbi,
    validate_window,
)


def make_prepared(count=5):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    graph = nx.MultiDiGraph()
    candidates = []
    trajectory = []
    base_x, base_y = transformer.transform(116.3, 39.9)
    for index in range(count):
        point_number = index + 1
        x = base_x + index * 15
        graph.add_node(index, x=x, y=base_y)
        graph.add_node(index + 1_000, x=x + 10, y=base_y)
        geometry = LineString([(x, base_y), (x + 10, base_y)])
        graph.add_edge(index, index + 1_000, key=0, geometry=geometry, length=10)
        trajectory.append({
            "point_number": point_number, "latitude": 39.9,
            "longitude": 116.3 + index * 0.0001, "timestamp": f"2020-01-01 00:00:{index:02d}",
            "segment_id": "trip",
        })
        candidates.append({
            "point_number": point_number, "u": index, "v": index + 1_000, "key": 0,
            "distance_meters": 2.0, "emission_probability": 1.0,
            "road_name": "test", "road_type": "residential", "maxspeed": None,
        })
    return {
        "trajectory": pd.DataFrame(trajectory), "candidates": pd.DataFrame(candidates),
        "graph": graph, "to_projected": transformer, "preprocessing_seconds": 0.0,
    }


class SlidingWindowTests(unittest.TestCase):
    def test_window_requires_buffer_less_than_window(self):
        self.assertEqual(validate_window(5, 2), (5, 2))
        for args in ((0, 0), (3, 3), (3, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_window(*args)

    def test_delayed_commitment_and_segment_end_flush(self):
        result = run_sliding_window_viterbi(make_prepared(), window_size=4, buffer_size=1)["observations"]
        self.assertEqual(len(result), 5)
        self.assertTrue(result.matched.all())
        self.assertEqual(list(result.prediction_delay_observations), [1, 1, 1, 1, 0])
        self.assertTrue((result.finalized_at_point >= result.point_number).all())

    def test_nearest_baseline_has_zero_delay(self):
        result = nearest_candidate_predictions(make_prepared())
        self.assertEqual(list(result.prediction_delay_observations), [0] * 5)

    def test_exact_edge_ground_truth_only_from_explicit_labels(self):
        unlabeled, count, _ = ground_truth_labels(pd.DataFrame({"latitude": [1]}))
        self.assertEqual((unlabeled, count), ({}, 0))
        labels, count, source = ground_truth_labels(pd.DataFrame({"truth": ["(1, 2, 0)", None]}), "truth")
        self.assertEqual(labels, {1: ("1", "2", "0")})
        self.assertEqual(count, 1)
        self.assertEqual(source, "truth")

    def test_experiment_outputs_csv_config_and_measured_plots(self):
        prepared = make_prepared(3)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / "results") as folder, patch(
            "src.evaluation.delay_accuracy.generate_trajectory_candidates", return_value=prepared
        ), patch("src.evaluation.delay_accuracy.decode_network_hmm") as full:
            full.return_value = {"observations": nearest_candidate_predictions(prepared), "decoding_seconds": .01}
            outcome = run_delay_experiment(
                [("synthetic", prepared["trajectory"])], window_sizes=[3], buffer_sizes=[0, 1],
                output_dir=folder, graph_context=(prepared["graph"], None, None, None),
            )
            self.assertTrue((Path(folder) / "experiment_results.csv").is_file())
            self.assertTrue((Path(folder) / "experiment_config.json").is_file())
            self.assertIn("coverage_by_buffer.png", outcome["plots"])
            self.assertNotIn("accuracy_by_buffer.png", outcome["plots"])
            self.assertTrue((Path(folder) / "ground_truth_accuracy_unavailable.txt").is_file())
            self.assertTrue(outcome["results"].ground_truth_accuracy_percent.isna().all())


if __name__ == "__main__":
    unittest.main()
