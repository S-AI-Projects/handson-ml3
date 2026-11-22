"""
Unit tests for R33-T035: Make preprocessing explicit
=====================================================
Covers:
  - MinMaxScaler: fit, transform, fit_transform
  - Contamination guard: transform() before fit() raises RuntimeError
  - Constant-feature rejection
  - Nearest-centroid assignments on the synthetic fixture
  - Full pipeline on the fixture
  - Deliberate wrong-answer detection (test must FAIL if expected values change)

All tests use standard library only. No numpy or sklearn.
"""

import math
import sys
import unittest

sys.path.insert(0, "course-labs/r33-t035")
from pipeline import (
    EXPECTED_ASSIGNMENTS,
    FIXTURE,
    FIXED_CENTROIDS,
    MinMaxScaler,
    euclidean,
    nearest_centroid_assignments,
    run_pipeline,
)


class TestMinMaxScaler(unittest.TestCase):

    def _make_rows(self, coords):
        return [{"record_id": f"r{i}", "x1": x, "x2": y}
                for i, (x, y) in enumerate(coords)]

    # ------------------------------------------------------------------
    # Basic fit + transform
    # ------------------------------------------------------------------
    def test_fit_stores_parameters(self):
        rows = self._make_rows([(0, 0), (6, 6)])
        scaler = MinMaxScaler()
        scaler.fit(rows)
        self.assertTrue(scaler.is_fitted)
        self.assertAlmostEqual(scaler._min[0], 0.0)
        self.assertAlmostEqual(scaler._max[0], 6.0)

    def test_transform_scales_correctly(self):
        rows = self._make_rows([(0, 0), (6, 6)])
        scaler = MinMaxScaler()
        scaler.fit(rows)
        scaled = scaler.transform(rows)
        self.assertEqual(len(scaled), 2)
        self.assertAlmostEqual(scaled[0][0], 0.0)
        self.assertAlmostEqual(scaled[0][1], 0.0)
        self.assertAlmostEqual(scaled[1][0], 1.0)
        self.assertAlmostEqual(scaled[1][1], 1.0)

    def test_fit_transform_matches_separate_calls(self):
        rows = self._make_rows([(1, 2), (3, 4), (5, 6)])
        s1 = MinMaxScaler()
        combined = s1.fit_transform(rows)
        s2 = MinMaxScaler()
        s2.fit(rows)
        separate = s2.transform(rows)
        for a, b in zip(combined, separate):
            self.assertAlmostEqual(a[0], b[0])
            self.assertAlmostEqual(a[1], b[1])

    # ------------------------------------------------------------------
    # Contamination guard
    # ------------------------------------------------------------------
    def test_transform_before_fit_raises(self):
        """Calling transform() before fit() must raise RuntimeError."""
        rows = self._make_rows([(1, 2)])
        scaler = MinMaxScaler()
        with self.assertRaises(RuntimeError):
            scaler.transform(rows)

    def test_fit_on_empty_raises(self):
        scaler = MinMaxScaler()
        with self.assertRaises(ValueError):
            scaler.fit([])

    def test_constant_feature_rejected(self):
        """A column with zero range must raise ValueError."""
        rows = self._make_rows([(3, 0), (3, 5)])  # x1 constant
        scaler = MinMaxScaler()
        with self.assertRaises(ValueError):
            scaler.fit(rows)

    # ------------------------------------------------------------------
    # Held-out transform uses only fitted params (no re-fit)
    # ------------------------------------------------------------------
    def test_held_out_transform_uses_train_params(self):
        train = self._make_rows([(0, 0), (6, 6)])
        test = self._make_rows([(3, 3)])
        scaler = MinMaxScaler()
        scaler.fit(train)
        scaled = scaler.transform(test)
        # 3/6 = 0.5 for both features
        self.assertAlmostEqual(scaled[0][0], 0.5)
        self.assertAlmostEqual(scaled[0][1], 0.5)


class TestEuclidean(unittest.TestCase):

    def test_zero_distance(self):
        self.assertAlmostEqual(euclidean((3, 4), (3, 4)), 0.0)

    def test_known_distance(self):
        self.assertAlmostEqual(euclidean((0, 0), (3, 4)), 5.0)


class TestNearestCentroid(unittest.TestCase):

    def test_fixture_assignments(self):
        """Core fixture: 6 points, 2 centroids → [0,0,0,1,1,1]."""
        points = [(r["x1"], r["x2"]) for r in FIXTURE]
        result = nearest_centroid_assignments(points, FIXED_CENTROIDS)
        self.assertEqual(result, EXPECTED_ASSIGNMENTS)

    def test_empty_centroids_raises(self):
        with self.assertRaises(ValueError):
            nearest_centroid_assignments([(1, 2)], [])

    def test_single_centroid(self):
        """All points assigned to 0 when only one centroid."""
        result = nearest_centroid_assignments([(0, 0), (10, 10)], [(5, 5)])
        self.assertEqual(result, [0, 0])

    def test_deliberate_wrong_answer_detected(self):
        """Test fails if expected assignments are tampered with."""
        points = [(r["x1"], r["x2"]) for r in FIXTURE]
        result = nearest_centroid_assignments(points, FIXED_CENTROIDS)
        wrong = [1, 1, 1, 0, 0, 0]  # deliberately reversed
        self.assertNotEqual(result, wrong,
                            "Tampered expected values were not detected.")


class TestPipeline(unittest.TestCase):

    def test_full_pipeline_on_fixture(self):
        """End-to-end: fixture rows → assignments == [0,0,0,1,1,1]."""
        result = run_pipeline(FIXTURE, FIXTURE, FIXED_CENTROIDS)
        self.assertEqual(result["assignments"], EXPECTED_ASSIGNMENTS)

    def test_pipeline_record_ids_preserved(self):
        result = run_pipeline(FIXTURE, FIXTURE, FIXED_CENTROIDS)
        expected_ids = [r["record_id"] for r in FIXTURE]
        self.assertEqual(result["record_ids"], expected_ids)

    def test_pipeline_scaler_is_fitted(self):
        result = run_pipeline(FIXTURE, FIXTURE, FIXED_CENTROIDS)
        self.assertTrue(result["scaler"].is_fitted)

    def test_pipeline_scaled_values_in_range(self):
        """All scaled training values should be in [0, 1]."""
        result = run_pipeline(FIXTURE, FIXTURE, FIXED_CENTROIDS)
        for x, y in result["scaled_train"]:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_pipeline_shape(self):
        result = run_pipeline(FIXTURE, FIXTURE, FIXED_CENTROIDS)
        self.assertEqual(len(result["assignments"]), len(FIXTURE))
        self.assertEqual(len(result["scaled_test"]), len(FIXTURE))


if __name__ == "__main__":
    unittest.main()
