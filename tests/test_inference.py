"""Focused validation and integration checks for GPS map-matching inference."""

import unittest

import pandas as pd

from src.map_matching.inference import (
    load_road_network,
    normalize_trajectory,
    predict_single_point,
    predict_trajectory,
    validate_coordinates,
)


class CoordinateValidationTests(unittest.TestCase):
    def test_rejects_out_of_range_and_non_finite_coordinates(self):
        for latitude, longitude in [(91, 0), (0, 181), (float("nan"), 0), ("north", 1)]:
            with self.subTest(latitude=latitude, longitude=longitude):
                with self.assertRaises(ValueError):
                    validate_coordinates(latitude, longitude)

    def test_normalizes_custom_column_names_and_preserves_segments(self):
        source = pd.DataFrame({
            "Y": [39.9, 39.91], "X": [116.3, 116.31],
            "when": ["2020-01-01", "2020-01-02"], "trip": ["a", "b"],
        })
        result = normalize_trajectory(
            source, latitude_column="Y", longitude_column="X",
            timestamp_column="when", segment_column="trip",
        )
        self.assertEqual(list(result.segment_id), ["a", "b"])
        self.assertEqual(list(result.point_number), [1, 2])
        self.assertEqual(list(result.latitude), [39.9, 39.91])

    def test_missing_coordinate_column_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "latitude column"):
            normalize_trajectory(pd.DataFrame({"x": [1], "longitude": [2]}))


class SavedGraphIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = load_road_network()

    def test_single_point_returns_real_nearby_edges(self):
        candidates = predict_single_point(
            39.984702, 116.318417, graph_context=self.network
        )
        self.assertFalse(candidates.empty)
        self.assertTrue(candidates.distance_meters.ge(0).all())
        self.assertAlmostEqual(float(candidates.emission_probability.sum()), 1.0, places=6)
        self.assertTrue(candidates.distance_meters.is_monotonic_increasing)

    def test_viterbi_pipeline_uses_sample_points_and_returns_unmatched_rows(self):
        from pathlib import Path

        sample = Path(__file__).resolve().parents[1] / "data" / "processed" / "user_000" / "20081023025304.csv"
        frame = pd.read_csv(sample).head(8)
        result = predict_trajectory(frame, graph_context=self.network)
        output = result["observations"]
        self.assertEqual(len(output), len(frame))
        self.assertTrue(output.matched.any())
        self.assertIn("sequence_block", output.columns)
        self.assertTrue(output.loc[output.matched, "distance_meters"].notna().all())
        self.assertGreaterEqual(result["runtime_seconds"], 0)

    def test_missing_candidate_splits_a_sequence_block(self):
        from pathlib import Path

        sample = Path(__file__).resolve().parents[1] / "data" / "processed" / "user_000" / "20081023025304.csv"
        frame = pd.read_csv(sample).head(8).copy()
        frame.loc[frame.index[3], "latitude"] = 40.5  # outside the saved Beijing graph
        result = predict_trajectory(frame, graph_context=self.network)
        output = result["observations"]
        self.assertEqual(int((~output.matched).sum()), 1)
        before = output.loc[output.point_number == 3, "sequence_block"].iloc[0]
        after = output.loc[output.point_number == 5, "sequence_block"].iloc[0]
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
