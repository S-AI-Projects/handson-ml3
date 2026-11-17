"""
Unit test suite for Course Lab R33-T001 schema validator & stratified split.
Validates input contract: record_id, group_id, x1, x2, label.
"""

from pathlib import Path
import unittest

from pipeline import (
    SchemaValidator,
    ValidationError,
    fixed_stratified_split,
    load_dataset,
)


class TestR33T001DataContract(unittest.TestCase):
    """Offline, CPU-only unit tests for classification input contract."""

    def setUp(self) -> None:
        self.validator = SchemaValidator()
        self.fixture_dir = Path(__file__).parent
        self.json_fixture_path = self.fixture_dir / "fixture.json"
        self.csv_fixture_path = self.fixture_dir / "fixture.csv"
        self.valid_fixture = load_dataset(self.json_fixture_path)

    # 1. Exact Synthetic Fixture Validation
    def test_fixture_exact_dimensions_and_values(self) -> None:
        """The synthetic fixture must have 8 rows with specified x1, x2, and label sequences."""
        self.assertEqual(len(self.valid_fixture), 8, "Fixture must contain exactly 8 synthetic rows.")

        expected_x1 = [0, 1, 2, 3, 4, 5, 6, 7]
        expected_x2 = [1, 1, 1, 1, 1, 1, 1, 1]
        expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]

        actual_x1 = [r["x1"] for r in self.valid_fixture]
        actual_x2 = [r["x2"] for r in self.valid_fixture]
        actual_labels = [r["label"] for r in self.valid_fixture]

        self.assertEqual(actual_x1, expected_x1)
        self.assertEqual(actual_x2, expected_x2)
        self.assertEqual(actual_labels, expected_labels)

        # Check stable record_ids and group_ids
        for r in self.valid_fixture:
            self.assertIn("record_id", r)
            self.assertIn("group_id", r)
            self.assertTrue(str(r["record_id"]).startswith("rec_"))
            self.assertTrue(str(r["group_id"]).startswith("grp_"))

    def test_valid_fixture_passes_without_errors(self) -> None:
        """The valid 8-row fixture must pass schema validation with zero errors."""
        valid_records, errors = self.validator.validate_records(self.valid_fixture)
        self.assertEqual(len(errors), 0, f"Expected zero errors on clean fixture, got: {errors}")
        self.assertEqual(len(valid_records), 8)

    def test_csv_fixture_loading_and_validation(self) -> None:
        """The CSV representation of the fixture must also load and validate identically."""
        csv_records = load_dataset(self.csv_fixture_path)
        self.assertEqual(len(csv_records), 8)
        valid_records, errors = self.validator.validate_records(csv_records)
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(valid_records), 8)

    # 2. Failure Cases: Missing ID
    def test_missing_or_empty_record_id_fails(self) -> None:
        """Missing or whitespace-only record_id must fail and report the field and reason."""
        invalid_rows = [
            {"record_id": None, "group_id": "grp_0", "x1": 1, "x2": 1, "label": 0},
            {"record_id": "   ", "group_id": "grp_0", "x1": 2, "x2": 1, "label": 0},
            {"group_id": "grp_0", "x1": 3, "x2": 1, "label": 0},
        ]
        valid_records, errors = self.validator.validate_records(invalid_rows)
        self.assertEqual(len(valid_records), 0)
        self.assertEqual(len(errors), 3)

        for err in errors:
            self.assertEqual(err.field, "record_id")
            self.assertIn("Missing or empty record_id", err.reason)

    # 3. Failure Cases: Duplicate Record ID
    def test_duplicate_record_id_fails(self) -> None:
        """Duplicate record_ids must report duplicate failure without dropping earlier valid rows silently."""
        records = [
            {"record_id": "dup_01", "group_id": "grp_0", "x1": 0, "x2": 1, "label": 0},
            {"record_id": "dup_01", "group_id": "grp_1", "x1": 4, "x2": 1, "label": 1},
        ]
        valid_records, errors = self.validator.validate_records(records)
        self.assertEqual(len(valid_records), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].field, "record_id")
        self.assertIn("Duplicate record_id: dup_01", errors[0].reason)

    # 4. Failure Cases: Missing Group ID
    def test_missing_group_id_fails(self) -> None:
        """Missing or empty group_id must be flagged."""
        records = [
            {"record_id": "rec_100", "group_id": None, "x1": 1, "x2": 1, "label": 0},
            {"record_id": "rec_101", "group_id": "", "x1": 2, "x2": 1, "label": 0},
        ]
        valid_records, errors = self.validator.validate_records(records)
        self.assertEqual(len(valid_records), 0)
        self.assertEqual(len(errors), 2)
        for err in errors:
            self.assertEqual(err.field, "group_id")
            self.assertIn("Missing or empty group_id", err.reason)

    # 5. Failure Cases: Malformed Numeric, Text, and Tensor Data
    def test_malformed_numeric_and_tensor_data_fails(self) -> None:
        """Strings, lists/tensors, NaNs, and booleans in x1 or x2 must be rejected with reasons."""
        malformed_records = [
            # String instead of float/int
            {"record_id": "bad_str", "group_id": "grp_0", "x1": "invalid_num", "x2": 1, "label": 0},
            # Tensor / list instead of scalar numeric
            {"record_id": "bad_tensor", "group_id": "grp_0", "x1": 1, "x2": [1.0, 2.0], "label": 0},
            # NaN value
            {"record_id": "bad_nan", "group_id": "grp_0", "x1": float("nan"), "x2": 1, "label": 0},
            # Boolean masquerading as number
            {"record_id": "bad_bool", "group_id": "grp_0", "x1": True, "x2": 1, "label": 0},
        ]
        valid_records, errors = self.validator.validate_records(malformed_records)
        self.assertEqual(len(valid_records), 0)
        self.assertEqual(len(errors), 4)

        error_fields = [e.field for e in errors]
        self.assertTrue(all(f in ("x1", "x2") for f in error_fields))
        self.assertTrue(any("must be int or float" in e.reason for e in errors))
        self.assertTrue(any("must be finite and non-NaN" in e.reason for e in errors))

    # 6. Failure Cases: Invalid Labels
    def test_invalid_label_fails(self) -> None:
        """Labels other than binary integer 0 or 1 must fail."""
        invalid_label_records = [
            {"record_id": "bad_lbl_1", "group_id": "grp_0", "x1": 1, "x2": 1, "label": 2},
            {"record_id": "bad_lbl_2", "group_id": "grp_0", "x1": 1, "x2": 1, "label": -1},
            {"record_id": "bad_lbl_3", "group_id": "grp_0", "x1": 1, "x2": 1, "label": "positive"},
            {"record_id": "bad_lbl_4", "group_id": "grp_0", "x1": 1, "x2": 1, "label": True},
        ]
        valid_records, errors = self.validator.validate_records(invalid_label_records)
        self.assertEqual(len(valid_records), 0)
        self.assertEqual(len(errors), 4)
        for err in errors:
            self.assertEqual(err.field, "label")
            self.assertIn("Invalid label", err.reason)

    # 7. Stratified Split Logic
    def test_fixed_stratified_split(self) -> None:
        """Stratified split on 8-row fixture must yield exactly balanced train and test sets."""
        train, test = fixed_stratified_split(self.valid_fixture, train_ratio=0.5)
        self.assertEqual(len(train), 4)
        self.assertEqual(len(test), 4)

        train_labels = [r["label"] for r in train]
        test_labels = [r["label"] for r in test]

        self.assertEqual(train_labels.count(0), 2)
        self.assertEqual(train_labels.count(1), 2)
        self.assertEqual(test_labels.count(0), 2)
        self.assertEqual(test_labels.count(1), 2)


if __name__ == "__main__":
    unittest.main()
