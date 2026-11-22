"""
Unit test suite for Course Lab R33-T034: Evaluation Metric Correctness.

Validates:
- Exact required classification case: truth[1,1,0,0]/pred[1,0,1,0] -> precision=0.5, recall=0.5.
- Exact required regression case: truth[1,2,3]/pred[1,3,2] -> MAE = 2/3.
- Explicit zero-denominator behavior for precision and recall.
- Exact 6-record synthetic text fixture (t1..t6 with specified strings and labels).
- Text fixture schema validation and reject rules.
- Deliberate mutation failure detection.
"""

from pathlib import Path
import copy
import unittest

from pipeline import (
    SchemaValidator,
    calculate_accuracy,
    calculate_confusion_matrix,
    calculate_f1_score,
    calculate_mae,
    calculate_precision,
    calculate_recall,
    calculate_token_exact_match,
    load_dataset,
)


class TestR33T034MetricCorrectness(unittest.TestCase):
    """Offline, CPU-only unit tests for metric correctness and text fixture."""

    def setUp(self) -> None:
        self.validator = SchemaValidator()
        self.fixture_dir = Path(__file__).parent
        self.json_fixture_path = self.fixture_dir / "fixture.json"
        self.csv_fixture_path = self.fixture_dir / "fixture.csv"
        self.fixture_records = load_dataset(self.json_fixture_path)

    # 1. Exact Mandated Classification Metric Test Case
    def test_mandated_classification_precision_recall_case(self) -> None:
        """For classification, truth[1,1,0,0]/prediction[1,0,1,0] must yield precision=0.5, recall=0.5."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]

        cm = calculate_confusion_matrix(y_true, y_pred)
        self.assertEqual(cm["TP"], 1)
        self.assertEqual(cm["FP"], 1)
        self.assertEqual(cm["FN"], 1)
        self.assertEqual(cm["TN"], 1)

        precision = calculate_precision(y_true, y_pred)
        recall = calculate_recall(y_true, y_pred)
        f1 = calculate_f1_score(y_true, y_pred)
        accuracy = calculate_accuracy(y_true, y_pred)

        self.assertAlmostEqual(precision, 0.5, places=6, msg="Precision must equal 0.5.")
        self.assertAlmostEqual(recall, 0.5, places=6, msg="Recall must equal 0.5.")
        self.assertAlmostEqual(f1, 0.5, places=6, msg="F1 must equal 0.5.")
        self.assertAlmostEqual(accuracy, 0.5, places=6, msg="Accuracy must equal 0.5.")

    # 2. Exact Mandated Regression Metric Test Case
    def test_mandated_regression_mae_case(self) -> None:
        """For regression, truth[1,2,3]/prediction[1,3,2] must yield MAE = 2/3."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 3.0, 2.0]

        mae = calculate_mae(y_true, y_pred)
        expected_mae = 2.0 / 3.0
        self.assertAlmostEqual(mae, expected_mae, places=6, msg="MAE must equal 2/3.")

    # 3. Explicit Zero-Denominator Handling
    def test_zero_denominator_precision_and_recall(self) -> None:
        """Zero positive predictions or zero positive truths must gracefully return zero_division without ZeroDivisionError."""
        # Case A: No positive predictions (TP + FP == 0)
        y_true = [1, 1, 0]
        y_pred = [0, 0, 0]
        p_default = calculate_precision(y_true, y_pred, zero_division=0.0)
        p_custom = calculate_precision(y_true, y_pred, zero_division=1.0)
        self.assertEqual(p_default, 0.0)
        self.assertEqual(p_custom, 1.0)

        # Case B: No positive ground truths (TP + FN == 0)
        y_true_zero = [0, 0, 0]
        y_pred_pos = [1, 1, 0]
        r_default = calculate_recall(y_true_zero, y_pred_pos, zero_division=0.0)
        r_custom = calculate_recall(y_true_zero, y_pred_pos, zero_division=1.0)
        self.assertEqual(r_default, 0.0)
        self.assertEqual(r_custom, 1.0)

        # Case C: F1 with all zeros
        f1_zero = calculate_f1_score([0, 0], [0, 0], zero_division=0.0)
        self.assertEqual(f1_zero, 0.0)

    # 4. Text Generation Exact Match Metric
    def test_text_exact_match_metric(self) -> None:
        """Text generation modality metric computes exact string match ratio."""
        true_texts = ["red apple", "blue sky", "green tree"]
        pred_texts = ["red apple", "dark sky", "green tree"]
        em = calculate_token_exact_match(true_texts, pred_texts)
        self.assertAlmostEqual(em, 2.0 / 3.0, places=6)

    # 5. Exact Synthetic Text Fixture Verification
    def test_synthetic_fixture_exact_records(self) -> None:
        """Fixture must contain exactly the 6 mandated records with IDs t1..t6 and specified texts and labels."""
        self.assertEqual(len(self.fixture_records), 6, "Must contain exactly 6 records.")

        expected_ids = ["t1", "t2", "t3", "t4", "t5", "t6"]
        expected_texts = ["red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"]
        expected_labels = [1, 1, 0, 0, 1, 0]

        actual_ids = [r["record_id"] for r in self.fixture_records]
        actual_texts = [r["text"] for r in self.fixture_records]
        actual_labels = [r["label"] for r in self.fixture_records]

        self.assertEqual(actual_ids, expected_ids, "Record IDs must match t1..t6.")
        self.assertEqual(actual_texts, expected_texts, "Text entries must match exactly.")
        self.assertEqual(actual_labels, expected_labels, "Labels must match [1,1,0,0,1,0].")

    def test_csv_matches_json_fixture(self) -> None:
        """CSV and JSON fixtures must represent identical records."""
        csv_records = load_dataset(self.csv_fixture_path)
        self.assertEqual(len(csv_records), len(self.fixture_records))
        for r_csv, r_json in zip(csv_records, self.fixture_records):
            self.assertEqual(r_csv["record_id"], r_json["record_id"])
            self.assertEqual(r_csv["text"], r_json["text"])
            self.assertEqual(r_csv["label"], r_json["label"])

    # 6. Schema Validation Pass & Rejection Tests
    def test_fixture_schema_validation(self) -> None:
        """Valid fixture passes schema validation with zero errors."""
        valid_recs, errors = self.validator.validate_records(self.fixture_records)
        self.assertEqual(len(valid_recs), 6)
        self.assertEqual(len(errors), 0, f"Clean fixture produced errors: {errors}")

    def test_schema_validator_rejects_malformed_records(self) -> None:
        """Empty text, duplicate ID, or invalid label must be rejected with descriptive ValidationError."""
        # Empty text
        bad_text = copy.deepcopy(self.fixture_records[0])
        bad_text["text"] = "   "
        errs = self.validator.validate_record(bad_text)
        self.assertTrue(any(e.field == "text" and "Empty" in e.reason for e in errs))

        # Duplicate ID
        dup_records = copy.deepcopy(self.fixture_records)
        dup_records[1]["record_id"] = "t1"
        _, dup_errs = self.validator.validate_records(dup_records)
        self.assertTrue(any(e.field == "record_id" and "Duplicate" in e.reason for e in dup_errs))

        # Invalid label
        bad_label = copy.deepcopy(self.fixture_records[0])
        bad_label["label"] = 99
        errs = self.validator.validate_record(bad_label)
        self.assertTrue(any(e.field == "label" and "out of range" in e.reason for e in errs))

    # 7. Deliberate Mutation Failure Detection
    def test_deliberate_mutation_detection(self) -> None:
        """Deliberately mutating expected MAE from 2/3 to 0.0 must raise an AssertionError."""
        mae = calculate_mae([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
        with self.assertRaises(AssertionError):
            self.assertEqual(mae, 0.0, "Deliberate assertion mutation must fail.")


if __name__ == "__main__":
    unittest.main()
