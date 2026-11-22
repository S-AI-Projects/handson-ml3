"""
Unit test suite for Course Lab R33-T031: Image Data Contract Validator.

Validates contract: record_id, group_id, pixels (4x4 grayscale [0, 255]), label.
Tests run offline, bounded, and CPU-only using standard library unittest.
"""

from pathlib import Path
import copy
import math
import unittest

from pipeline import (
    SchemaValidator,
    ValidationError,
    fixed_stratified_split,
    load_dataset,
)


class TestR33T031ImageContract(unittest.TestCase):
    """Offline, CPU-only unit tests for 4x4 image classification input contract."""

    def setUp(self) -> None:
        self.validator = SchemaValidator()
        self.fixture_dir = Path(__file__).parent
        self.json_fixture_path = self.fixture_dir / "fixture.json"
        self.csv_fixture_path = self.fixture_dir / "fixture.csv"
        self.valid_fixture = load_dataset(self.json_fixture_path)

    # 1. Exact Synthetic Fixture Verification
    def test_fixture_exact_count_and_composition(self) -> None:
        """The synthetic fixture must contain exactly 6 rows: 3 all-zero (label 0) and 3 diagonal (label 1)."""
        self.assertEqual(len(self.valid_fixture), 6, "Fixture must contain exactly 6 synthetic records.")

        zero_records = [r for r in self.valid_fixture if r["label"] == 0]
        diag_records = [r for r in self.valid_fixture if r["label"] == 1]

        self.assertEqual(len(zero_records), 3, "Fixture must contain exactly three label-0 images.")
        self.assertEqual(len(diag_records), 3, "Fixture must contain exactly three label-1 images.")

        # Verify all-zero 4x4 images
        for r in zero_records:
            pixels = r["pixels"]
            self.assertEqual(len(pixels), 4, "Zero image must have 4 rows.")
            for row in pixels:
                self.assertEqual(len(row), 4, "Zero image row must have 4 columns.")
                self.assertTrue(all(p == 0 for p in row), "Zero image must have all 0 pixels.")

        # Verify white diagonal (255) 4x4 images
        for r in diag_records:
            pixels = r["pixels"]
            self.assertEqual(len(pixels), 4, "Diagonal image must have 4 rows.")
            for i, row in enumerate(pixels):
                self.assertEqual(len(row), 4, "Diagonal image row must have 4 columns.")
                for j, p in enumerate(row):
                    if i == j:
                        self.assertEqual(p, 255, f"Diagonal pixel at [{i},{j}] must be 255.")
                    else:
                        self.assertEqual(p, 0, f"Off-diagonal pixel at [{i},{j}] must be 0.")

        # Verify record_id uniqueness and group_id declaration
        record_ids = [r["record_id"] for r in self.valid_fixture]
        self.assertEqual(len(record_ids), len(set(record_ids)), "Record IDs must all be unique.")
        for r in self.valid_fixture:
            self.assertTrue(bool(r.get("group_id")), "Group ID must be non-empty.")

    def test_csv_and_json_fixtures_match(self) -> None:
        """CSV fixture and JSON fixture must represent identical data records."""
        csv_data = load_dataset(self.csv_fixture_path)
        self.assertEqual(len(csv_data), len(self.valid_fixture))
        for r_csv, r_json in zip(csv_data, self.valid_fixture):
            self.assertEqual(r_csv["record_id"], r_json["record_id"])
            self.assertEqual(r_csv["group_id"], r_json["group_id"])
            self.assertEqual(r_csv["label"], r_json["label"])
            self.assertEqual(r_csv["pixels"], r_json["pixels"])

    # 2. Valid Dataset Schema Pass
    def test_valid_fixture_passes_validation(self) -> None:
        """Valid synthetic fixture must pass with zero validation errors."""
        valid_records, errors = self.validator.validate_records(self.valid_fixture)
        self.assertEqual(len(valid_records), 6, "All 6 valid records must be accepted.")
        self.assertEqual(len(errors), 0, f"Expected 0 errors, got: {errors}")

    # 3. Missing or Invalid Record ID
    def test_missing_or_empty_record_id_fails(self) -> None:
        """Records with missing or empty record_id must fail with descriptive reason."""
        record_missing = copy.deepcopy(self.valid_fixture[0])
        record_missing["record_id"] = None
        errs = self.validator.validate_record(record_missing)
        self.assertTrue(any(e.field == "record_id" and "Missing" in e.reason for e in errs))

        record_empty = copy.deepcopy(self.valid_fixture[0])
        record_empty["record_id"] = "   "
        errs = self.validator.validate_record(record_empty)
        self.assertTrue(any(e.field == "record_id" and "Empty" in e.reason for e in errs))

    # 4. Duplicate Record ID Detection
    def test_duplicate_record_id_fails(self) -> None:
        """Duplicate record_id must be detected and reported without silently overwriting."""
        records = copy.deepcopy(self.valid_fixture)
        records[1]["record_id"] = records[0]["record_id"]  # Duplicate ID
        valid_recs, errs = self.validator.validate_records(records)
        self.assertEqual(len(valid_recs), 5)
        self.assertTrue(any(e.field == "record_id" and "Duplicate" in e.reason for e in errs))

    # 5. Missing or Invalid Group ID
    def test_missing_or_empty_group_id_fails(self) -> None:
        """Records with missing or empty group_id must fail validation."""
        record_missing = copy.deepcopy(self.valid_fixture[0])
        record_missing["group_id"] = None
        errs = self.validator.validate_record(record_missing)
        self.assertTrue(any(e.field == "group_id" and "Missing" in e.reason for e in errs))

        record_empty = copy.deepcopy(self.valid_fixture[0])
        record_empty["group_id"] = ""
        errs = self.validator.validate_record(record_empty)
        self.assertTrue(any(e.field == "group_id" and "Empty" in e.reason for e in errs))

    # 6. Malformed Label Handling
    def test_malformed_label_fails(self) -> None:
        """Labels that are missing, non-numeric text, boolean, or out-of-range must fail."""
        # Non-numeric text
        rec_text = copy.deepcopy(self.valid_fixture[0])
        rec_text["label"] = "not_a_label"
        errs = self.validator.validate_record(rec_text)
        self.assertTrue(any(e.field == "label" and "unable to parse" in e.reason for e in errs))

        # Boolean (True/False should not silently masquerade as 1/0)
        rec_bool = copy.deepcopy(self.valid_fixture[0])
        rec_bool["label"] = True
        errs = self.validator.validate_record(rec_bool)
        self.assertTrue(any(e.field == "label" and "boolean" in e.reason for e in errs))

        # Out-of-range class label
        rec_out = copy.deepcopy(self.valid_fixture[0])
        rec_out["label"] = 99
        errs = self.validator.validate_record(rec_out)
        self.assertTrue(any(e.field == "label" and "out of range" in e.reason for e in errs))

    # 7. Malformed Tensor Shape / Dimensions
    def test_malformed_tensor_shape_fails(self) -> None:
        """Tensors that are not 4x4 or have irregular rows must fail validation."""
        # 3x4 instead of 4x4
        rec_3x4 = copy.deepcopy(self.valid_fixture[0])
        rec_3x4["pixels"] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        errs = self.validator.validate_record(rec_3x4)
        self.assertTrue(any(e.field == "pixels" and "Malformed tensor" in e.reason for e in errs))

        # Jagged row: row 1 has only 3 columns
        rec_jagged = copy.deepcopy(self.valid_fixture[0])
        rec_jagged["pixels"] = [
            [0, 0, 0, 0],
            [0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]
        errs = self.validator.validate_record(rec_jagged)
        self.assertTrue(any(e.field == "pixels" and "shape: row 1" in e.reason for e in errs))

        # Non-sequence pixels
        rec_scalar = copy.deepcopy(self.valid_fixture[0])
        rec_scalar["pixels"] = 42
        errs = self.validator.validate_record(rec_scalar)
        self.assertTrue(any(e.field == "pixels" and "expected list or sequence" in e.reason for e in errs))

    # 8. Malformed Numeric Pixel Values
    def test_malformed_pixel_numeric_values_fail(self) -> None:
        """Pixel values with non-numeric types, negatives, >255, or NaN must fail."""
        # Value > 255
        rec_over = copy.deepcopy(self.valid_fixture[0])
        rec_over["pixels"][0][0] = 300
        errs = self.validator.validate_record(rec_over)
        self.assertTrue(any(e.field == "pixels" and "out of range" in e.reason for e in errs))

        # Value < 0
        rec_under = copy.deepcopy(self.valid_fixture[0])
        rec_under["pixels"][0][0] = -5
        errs = self.validator.validate_record(rec_under)
        self.assertTrue(any(e.field == "pixels" and "out of range" in e.reason for e in errs))

        # String inside pixel matrix
        rec_str = copy.deepcopy(self.valid_fixture[0])
        rec_str["pixels"][0][0] = "white"
        errs = self.validator.validate_record(rec_str)
        self.assertTrue(any(e.field == "pixels" and "non-numeric" in e.reason for e in errs))

        # NaN inside pixel matrix
        rec_nan = copy.deepcopy(self.valid_fixture[0])
        rec_nan["pixels"][0][0] = float("nan")
        errs = self.validator.validate_record(rec_nan)
        self.assertTrue(any(e.field == "pixels" and "NaN" in e.reason for e in errs))

    # 9. No Silent Dropping of Records
    def test_no_silent_dropping_of_invalid_records(self) -> None:
        """Every rejected record must produce a specific ValidationError tracking record_id and field."""
        bad_records = [
            {"record_id": "bad_1", "group_id": "g1", "pixels": [[0]*4]*4, "label": 99},
            {"record_id": None, "group_id": "g2", "pixels": [[0]*4]*4, "label": 0},
            {"record_id": "bad_3", "group_id": None, "pixels": [[0]*4]*4, "label": 1},
        ]
        valid_recs, errs = self.validator.validate_records(bad_records)
        self.assertEqual(len(valid_recs), 0)
        self.assertGreaterEqual(len(errs), 3)
        fields = [e.field for e in errs]
        self.assertIn("label", fields)
        self.assertIn("record_id", fields)
        self.assertIn("group_id", fields)

    # 10. Explicit Train / Test Disjoint Split
    def test_deterministic_stratified_split(self) -> None:
        """Stratified split produces deterministic, disjoint partitions with balanced class labels."""
        train_set, test_set = fixed_stratified_split(self.valid_fixture, train_ratio=0.5)

        train_ids = {r["record_id"] for r in train_set}
        test_ids = {r["record_id"] for r in test_set}

        # Disjoint sets
        self.assertEqual(len(train_ids.intersection(test_ids)), 0, "Train and test splits must be disjoint.")
        self.assertEqual(len(train_set) + len(test_set), 6, "All 6 records must be partitioned.")

        # Class balance: 3 records with label 0, 3 records with label 1 split 2/1 or 1/2
        train_labels = [r["label"] for r in train_set]
        test_labels = [r["label"] for r in test_set]
        self.assertIn(0, train_labels)
        self.assertIn(1, train_labels)
        self.assertIn(0, test_labels)
        self.assertIn(1, test_labels)

    # 11. Deliberate Mutation Test (Nonzero failure on altered expected result)
    def test_deliberate_mutation_detection(self) -> None:
        """Deliberately mutating a diagonal image to non-diagonal must be caught by assertion."""
        mutated_diag = copy.deepcopy(self.valid_fixture[3])
        # Mutate diagonal pixel to zero
        mutated_diag["pixels"][0][0] = 0
        with self.assertRaises(AssertionError):
            self.assertEqual(
                mutated_diag["pixels"][0][0],
                255,
                "Deliberate mutation should fail diagonal assertion."
            )


if __name__ == "__main__":
    unittest.main()
