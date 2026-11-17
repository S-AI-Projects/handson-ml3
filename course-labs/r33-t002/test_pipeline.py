"""
Unit test suite for Course Lab R33-T002 deterministic split manifest.
Tests disjointness, group isolation, chronological order, and fixture validity.
"""

from pathlib import Path
import unittest

from pipeline import (
    LeakageError,
    SchemaValidationError,
    SplitManifest,
    build_deterministic_split_manifest,
    load_dataset,
    validate_regression_schema,
)


class TestR33T002SplitManifest(unittest.TestCase):
    """Offline, CPU-only unit tests for regression split leakage prevention."""

    def setUp(self) -> None:
        self.lab_dir = Path(__file__).parent
        self.json_fixture_path = self.lab_dir / "fixture.json"
        self.csv_fixture_path = self.lab_dir / "fixture.csv"
        self.fixture = load_dataset(self.json_fixture_path)

    # 1. Exact Synthetic Fixture & Formula Verification
    def test_fixture_exact_dimensions_and_formula(self) -> None:
        """Fixture must contain 6 synthetic rows satisfying target = 2*x1 + 1."""
        self.assertEqual(len(self.fixture), 6)

        expected_x1 = [0, 1, 2, 3, 4, 5]
        expected_x2 = [1, 1, 1, 1, 1, 1]
        expected_targets = [2 * x + 1 for x in expected_x1]  # [1, 3, 5, 7, 9, 11]
        expected_ids = ["r1", "r2", "r3", "r4", "r5", "r6"]

        actual_x1 = [r["x1"] for r in self.fixture]
        actual_x2 = [r["x2"] for r in self.fixture]
        actual_targets = [r["target"] for r in self.fixture]
        actual_ids = [r["record_id"] for r in self.fixture]

        self.assertEqual(actual_x1, expected_x1)
        self.assertEqual(actual_x2, expected_x2)
        self.assertEqual(actual_targets, expected_targets)
        self.assertEqual(actual_ids, expected_ids)

    # 2. Manifest Creation & Disjointness
    def test_deterministic_manifest_on_clean_fixture(self) -> None:
        """Manifest builds cleanly and assigns r1..r4 to train, r5 to val, r6 to test."""
        manifest = build_deterministic_split_manifest(self.fixture)

        self.assertEqual(manifest.train_record_ids, ["r1", "r2", "r3", "r4"])
        self.assertEqual(manifest.validation_record_ids, ["r5"])
        self.assertEqual(manifest.test_record_ids, ["r6"])
        self.assertTrue(manifest.is_disjoint)
        self.assertEqual(manifest.counts["train"], 4)
        self.assertEqual(manifest.counts["validation"], 1)
        self.assertEqual(manifest.counts["test"], 1)

    def test_manifest_reproducibility(self) -> None:
        """Repeated runs must produce identical manifests."""
        run1 = build_deterministic_split_manifest(self.fixture).to_dict()
        run2 = build_deterministic_split_manifest(self.fixture).to_dict()
        self.assertEqual(run1, run2)

    def test_csv_fixture_loads_and_splits_identically(self) -> None:
        """CSV fixture must load and generate identical manifest."""
        csv_records = load_dataset(self.csv_fixture_path)
        manifest_csv = build_deterministic_split_manifest(csv_records)
        manifest_json = build_deterministic_split_manifest(self.fixture)
        self.assertEqual(manifest_csv.to_dict(), manifest_json.to_dict())

    # 3. Leakage Prevention: Record Overlap
    def test_overlapping_record_ids_rejected(self) -> None:
        """Intentional overlap between train and validation must raise LeakageError."""
        with self.assertRaises(LeakageError) as ctx:
            build_deterministic_split_manifest(
                self.fixture,
                train_ids=["r1", "r2", "r3", "r4"],
                validation_ids=["r4", "r5"],  # 'r4' intentionally overlaps!
                test_ids=["r6"],
            )
        self.assertIn("Split leakage detected: record sets are not disjoint", str(ctx.exception))

    def test_overlap_between_train_and_test_rejected(self) -> None:
        """Intentional overlap between train and test must raise LeakageError."""
        with self.assertRaises(LeakageError) as ctx:
            build_deterministic_split_manifest(
                self.fixture,
                train_ids=["r1", "r2", "r3", "r6"],  # 'r6' in both train and test
                validation_ids=["r5"],
                test_ids=["r6"],
            )
        self.assertIn("Split leakage detected", str(ctx.exception))

    # 4. Leakage Prevention: Group Overlap
    def test_overlapping_groups_rejected(self) -> None:
        """Group leakage across splits must raise LeakageError."""
        leaky_data = [
            {"record_id": "r1", "x1": 0, "x2": 1, "target": 1, "group_id": "patient_A"},
            {"record_id": "r2", "x1": 1, "x2": 1, "target": 3, "group_id": "patient_B"},
            {"record_id": "r5", "x1": 4, "x2": 1, "target": 9, "group_id": "patient_A"},  # same group in val!
            {"record_id": "r6", "x1": 5, "x2": 1, "target": 11, "group_id": "patient_C"},
        ]
        with self.assertRaises(LeakageError) as ctx:
            build_deterministic_split_manifest(
                leaky_data,
                train_ids=["r1", "r2"],
                validation_ids=["r5"],
                test_ids=["r6"],
            )
        self.assertIn("Group leakage detected", str(ctx.exception))

    # 5. Leakage Prevention: Chronological Time-Series
    def test_chronological_time_series_leakage_rejected(self) -> None:
        """Future observations leaking into the training set must raise LeakageError."""
        time_leaky_data = [
            {"record_id": "r1", "x1": 0, "x2": 1, "target": 1, "timestamp": 100},  # future timestamp!
            {"record_id": "r2", "x1": 1, "x2": 1, "target": 3, "timestamp": 2},
            {"record_id": "r3", "x1": 2, "x2": 1, "target": 5, "timestamp": 3},
            {"record_id": "r4", "x1": 3, "x2": 1, "target": 7, "timestamp": 4},
            {"record_id": "r5", "x1": 4, "x2": 1, "target": 9, "timestamp": 5},
            {"record_id": "r6", "x1": 5, "x2": 1, "target": 11, "timestamp": 6},
        ]
        with self.assertRaises(LeakageError) as ctx:
            build_deterministic_split_manifest(time_leaky_data)
        self.assertIn("Temporal leakage", str(ctx.exception))

    # 6. Schema & Type Errors
    def test_malformed_numeric_and_empty_id(self) -> None:
        """Malformed target or missing record_id must raise SchemaValidationError."""
        with self.assertRaises(SchemaValidationError):
            validate_regression_schema([{"record_id": "", "x1": 0, "x2": 1, "target": 1}])

        with self.assertRaises(SchemaValidationError):
            validate_regression_schema([{"record_id": "r1", "x1": "string_val", "x2": 1, "target": 1}])

        with self.assertRaises(SchemaValidationError):
            validate_regression_schema([{"record_id": "r1", "x1": 0, "x2": 1, "target": float("nan")}])


if __name__ == "__main__":
    unittest.main()
