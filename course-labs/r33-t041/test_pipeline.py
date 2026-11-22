"""
Unit tests for Course Lab R33-T041 schema validator and fixture checks.
"""
import unittest
import json
import os
from pipeline import validate_record, validate_dataset, partition_time_series

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestTimeSeriesDataContract(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)

    def test_valid_fixture_passes(self):
        result = validate_dataset(self.fixture)
        self.assertEqual(result["summary"]["total"], 10)
        self.assertEqual(result["summary"]["passed"], 10)
        self.assertEqual(result["summary"]["failed"], 0)
        self.assertEqual(len(result["failures"]), 0)

    def test_fixture_values_and_ids(self):
        expected_ids = [f"d{i}" for i in range(1, 11)]
        expected_vals = [10, 12, 11, 13, 15, 14, 16, 18, 17, 19]
        actual_ids = [r["record_id"] for r in self.fixture]
        actual_vals = [r["value"] for r in self.fixture]
        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(actual_vals, expected_vals)

    def test_partition_splits_no_shuffling(self):
        splits = partition_time_series(self.fixture)
        self.assertEqual([r["record_id"] for r in splits["train"]], ["d1", "d2", "d3", "d4", "d5", "d6"])
        self.assertEqual([r["record_id"] for r in splits["validation"]], ["d7", "d8"])
        self.assertEqual([r["record_id"] for r in splits["test"]], ["d9", "d10"])

    def test_missing_record_id_fails(self):
        bad_rec = {"timestamp": "2026-01-01", "value": 10}
        passed, err = validate_record(bad_rec)
        self.assertFalse(passed)
        self.assertEqual(err["field"], "record_id")
        self.assertIn("Missing or empty", err["reason"])

    def test_duplicate_record_id_fails(self):
        seen = {"d1"}
        bad_rec = {"record_id": "d1", "timestamp": "2026-01-02", "value": 12}
        passed, err = validate_record(bad_rec, seen_ids=seen)
        self.assertFalse(passed)
        self.assertEqual(err["field"], "record_id")
        self.assertIn("Duplicate", err["reason"])

    def test_malformed_numeric_string_fails(self):
        bad_rec = {"record_id": "d1", "timestamp": "2026-01-01", "value": "10"}
        passed, err = validate_record(bad_rec)
        self.assertFalse(passed)
        self.assertEqual(err["field"], "value")
        self.assertIn("Malformed numeric", err["reason"])

    def test_malformed_numeric_nan_fails(self):
        bad_rec = {"record_id": "d1", "timestamp": "2026-01-01", "value": float("nan")}
        passed, err = validate_record(bad_rec)
        self.assertFalse(passed)
        self.assertEqual(err["field"], "value")
        self.assertIn("NaN or Inf", err["reason"])

    def test_malformed_timestamp_fails(self):
        bad_rec = {"record_id": "d1", "timestamp": "01-01-2026", "value": 10}
        passed, err = validate_record(bad_rec)
        self.assertFalse(passed)
        self.assertEqual(err["field"], "timestamp")
        self.assertIn("Malformed timestamp", err["reason"])

    def test_tensor_list_value_fails(self):
        bad_rec = {"record_id": "d1", "timestamp": "2026-01-01", "value": [10]}
        passed, err = validate_record(bad_rec)
        self.assertFalse(passed)
        self.assertEqual(err["field"], "value")
        self.assertIn("Malformed numeric", err["reason"])

    def test_deliberate_change_detected(self):
        corrupted = list(self.fixture)
        corrupted[3] = {"record_id": "d4", "timestamp": "2026-01-04", "value": "invalid"}
        res = validate_dataset(corrupted)
        self.assertEqual(res["summary"]["failed"], 1)
        self.assertEqual(res["failures"][0]["record_id"], "d4")

if __name__ == "__main__":
    unittest.main()
