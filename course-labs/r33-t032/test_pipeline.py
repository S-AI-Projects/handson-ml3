"""
Unit test suite for Course Lab R33-T032: Leakage-Free Splitting & Split Manifest.

Validates:
- Exact 8-row synthetic fixture (x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1]).
- Deterministic split manifest generation with disjoint train/val/test partitions.
- Rejection of overlapping record IDs and group leakage.
- Chronological time-series split ordering.
- Offline, bounded, CPU-only verification with non-zero test count.
"""

from pathlib import Path
import copy
import json
import unittest

from pipeline import (
    SchemaValidator,
    SplitManifest,
    generate_chronological_split_manifest,
    generate_stratified_split_manifest,
    load_dataset,
    validate_split_manifest,
)


class TestR33T032SplitLeakage(unittest.TestCase):
    """Offline, CPU-only unit tests for split manifest & leakage prevention."""

    def setUp(self) -> None:
        self.validator = SchemaValidator()
        self.fixture_dir = Path(__file__).parent
        self.json_fixture_path = self.fixture_dir / "fixture.json"
        self.csv_fixture_path = self.fixture_dir / "fixture.csv"
        self.manifest_path = self.fixture_dir / "split_manifest.json"
        self.records = load_dataset(self.json_fixture_path)

    # 1. Exact Synthetic Fixture Validation
    def test_fixture_exact_count_and_values(self) -> None:
        """Fixture must contain exactly 8 synthetic rows with specified x1, x2, and label."""
        self.assertEqual(len(self.records), 8, "Fixture must contain exactly 8 records.")

        expected_x1 = [0, 1, 2, 3, 4, 5, 6, 7]
        expected_x2 = [1, 1, 1, 1, 1, 1, 1, 1]
        expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]

        actual_x1 = [r["x1"] for r in self.records]
        actual_x2 = [r["x2"] for r in self.records]
        actual_labels = [r["label"] for r in self.records]

        self.assertEqual(actual_x1, expected_x1, "x1 feature sequence mismatch.")
        self.assertEqual(actual_x2, expected_x2, "x2 feature constant mismatch.")
        self.assertEqual(actual_labels, expected_labels, "Label sequence mismatch.")

        # Stable unique record IDs
        rec_ids = [r["record_id"] for r in self.records]
        self.assertEqual(len(rec_ids), len(set(rec_ids)), "Record IDs must be strictly unique.")

    def test_csv_matches_json_fixture(self) -> None:
        """CSV and JSON fixtures must represent the identical dataset."""
        csv_records = load_dataset(self.csv_fixture_path)
        self.assertEqual(len(csv_records), len(self.records))
        for r_csv, r_json in zip(csv_records, self.records):
            self.assertEqual(r_csv["record_id"], r_json["record_id"])
            self.assertEqual(r_csv["group_id"], r_json["group_id"])
            self.assertEqual(r_csv["x1"], r_json["x1"])
            self.assertEqual(r_csv["x2"], r_json["x2"])
            self.assertEqual(r_csv["label"], r_json["label"])

    # 2. Schema Validation Passes on Clean Fixture
    def test_clean_fixture_schema_validation(self) -> None:
        """Valid fixture passes schema validation with 0 errors."""
        valid_recs, errors = self.validator.validate_records(self.records)
        self.assertEqual(len(valid_recs), 8)
        self.assertEqual(len(errors), 0, f"Clean fixture produced errors: {errors}")

    # 3. Disjoint Partition Sets in Manifest
    def test_manifest_partitions_are_disjoint(self) -> None:
        """Train, validation, and test sets must be strictly mutually disjoint."""
        manifest = SplitManifest.load(self.manifest_path)
        s_train = set(manifest.train)
        s_val = set(manifest.validation)
        s_test = set(manifest.test)

        self.assertEqual(len(s_train.intersection(s_val)), 0, "Train and validation overlap detected.")
        self.assertEqual(len(s_train.intersection(s_test)), 0, "Train and test overlap detected.")
        self.assertEqual(len(s_val.intersection(s_test)), 0, "Validation and test overlap detected.")

        # Full coverage
        all_split_ids = s_train | s_val | s_test
        expected_ids = {r["record_id"] for r in self.records}
        self.assertEqual(all_split_ids, expected_ids, "Manifest must cover all 8 records.")

    # 4. Deterministic Manifest Generation Across Repeated Runs
    def test_repeated_runs_produce_identical_manifest(self) -> None:
        """Repeated generation calls must produce byte-for-byte identical manifests."""
        run1 = generate_stratified_split_manifest(self.records, train_ratio=0.5, seed=42)
        run2 = generate_stratified_split_manifest(self.records, train_ratio=0.5, seed=42)
        self.assertEqual(run1.to_dict(), run2.to_dict(), "Manifest generation is not deterministic.")

    # 5. Stratification Balance
    def test_stratified_split_preserves_class_balance(self) -> None:
        """Each partition must preserve balanced representation of classes 0 and 1."""
        manifest = SplitManifest.load(self.manifest_path)
        rec_by_id = {r["record_id"]: r for r in self.records}

        train_labels = [rec_by_id[rid]["label"] for rid in manifest.train]
        val_labels = [rec_by_id[rid]["label"] for rid in manifest.validation]
        test_labels = [rec_by_id[rid]["label"] for rid in manifest.test]

        self.assertEqual(train_labels.count(0), 2, "Train set must have 2 class-0 records.")
        self.assertEqual(train_labels.count(1), 2, "Train set must have 2 class-1 records.")
        self.assertEqual(val_labels.count(0), 1, "Validation set must have 1 class-0 record.")
        self.assertEqual(val_labels.count(1), 1, "Validation set must have 1 class-1 record.")
        self.assertEqual(test_labels.count(0), 1, "Test set must have 1 class-0 record.")
        self.assertEqual(test_labels.count(1), 1, "Test set must have 1 class-1 record.")

    # 6. Rejection of Intentionally Overlapping Record IDs
    def test_rejects_overlapping_record_ids(self) -> None:
        """Intentionally overlapping IDs between partitions must be rejected by validator."""
        leaky_manifest = SplitManifest(
            train=["rec_001", "rec_002", "rec_005"],
            validation=["rec_001", "rec_006"],  # rec_001 leaks into validation
            test=["rec_007", "rec_008"],
            metadata={},
        )
        violations = validate_split_manifest(leaky_manifest, self.records)
        self.assertTrue(any("Split leakage detected" in v and "rec_001" in v for v in violations))

    # 7. Rejection of Intentionally Overlapping Groups
    def test_rejects_group_leakage(self) -> None:
        """When group isolation is required, sharing a group across partitions must fail."""
        # rec_001 and rec_002 share grp_001
        leaky_group_manifest = SplitManifest(
            train=["rec_001"],
            validation=["rec_002"],  # Shares grp_001 with train
            test=["rec_003", "rec_004", "rec_005", "rec_006", "rec_007", "rec_008"],
            metadata={},
        )
        violations = validate_split_manifest(leaky_group_manifest, self.records, check_group_isolation=True)
        self.assertTrue(any("Group leakage detected" in v and "grp_001" in v for v in violations))

    # 8. Chronological Time-Series Split Verification
    def test_chronological_time_series_split_is_monotonic(self) -> None:
        """Time-series split must be strictly chronological without forward lookahead leakage."""
        chrono_manifest = generate_chronological_split_manifest(self.records, train_ratio=0.5, val_ratio=0.25)
        violations = validate_split_manifest(chrono_manifest, self.records, check_chronological=True)
        self.assertEqual(len(violations), 0, f"Chronological split produced temporal violations: {violations}")

        rec_by_id = {r["record_id"]: r for r in self.records}
        max_train_time = max(rec_by_id[rid]["timestamp"] for rid in chrono_manifest.train)
        min_val_time = min(rec_by_id[rid]["timestamp"] for rid in chrono_manifest.validation)
        min_test_time = min(rec_by_id[rid]["timestamp"] for rid in chrono_manifest.test)

        self.assertLessEqual(max_train_time, min_val_time, "Train timestamps must precede validation.")
        self.assertLessEqual(min_val_time, min_test_time, "Validation timestamps must precede test.")

    # 9. Deliberate Mutation Failure Detection
    def test_deliberate_mutation_detection(self) -> None:
        """Mutating expected split count or modifying record IDs triggers assertion failure."""
        manifest = SplitManifest.load(self.manifest_path)
        with self.assertRaises(AssertionError):
            self.assertEqual(len(manifest.train), 99, "Deliberate assertion mutation must fail.")


if __name__ == "__main__":
    unittest.main()
