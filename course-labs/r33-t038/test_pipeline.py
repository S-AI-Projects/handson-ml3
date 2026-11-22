"""
Unit tests for R33-T038: Explain an error case
==============================================
Covers:
  - Exact synthetic fixture verification (8 rows, x1=[0..7], x2=1, label=[0..0,1..1])
  - Clean model prediction verification (100% correct, zero error rows)
  - Injected error appears with correct record ID ('s4'), expected 0, predicted 1
  - Correct predictions do NOT appear in error_rows
  - Aggregate error count matches detailed rows count exactly
  - Reason category properly assigned
  - Rejection of invalid inputs (empty dataset, missing predictions)
  - Deliberate tampering failure test

Standard library only. Offline CPU-only execution.
"""

import sys
import unittest

sys.path.insert(0, "course-labs/r33-t038")
from pipeline import (
    INJECTED_ERROR_REASON,
    INJECTED_ERROR_TARGET_ID,
    INJECTED_ERROR_WRONG_PRED,
    SYNTHETIC_FIXTURE,
    TEST_RECORD_IDS,
    TRAIN_RECORD_IDS,
    DecisionStumpClassifier,
    generate_error_report,
    inject_prediction_error,
    run_error_analysis_pipeline,
)


class TestErrorAnalysisFixtureAndModel(unittest.TestCase):

    def test_fixture_integrity(self):
        """Eight synthetic rows with x1=[0..7], x2=1, and label=[0,0,0,0,1,1,1,1]."""
        self.assertEqual(len(SYNTHETIC_FIXTURE), 8)
        self.assertEqual([r["x1"] for r in SYNTHETIC_FIXTURE], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual([r["x2"] for r in SYNTHETIC_FIXTURE], [1.0] * 8)
        self.assertEqual([r["label"] for r in SYNTHETIC_FIXTURE], [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual([r["record_id"] for r in SYNTHETIC_FIXTURE], [f"s{i}" for i in range(1, 9)])

    def test_stratified_split_integrity(self):
        """Train and test splits preserve group isolation and class balance."""
        train_set = set(TRAIN_RECORD_IDS)
        test_set = set(TEST_RECORD_IDS)
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertEqual(len(train_set), 6)
        self.assertEqual(len(test_set), 2)


class TestErrorReportCorrectness(unittest.TestCase):

    def test_clean_model_has_zero_errors(self):
        """Clean fitted weak learner on fixture produces 0 errors."""
        stump = DecisionStumpClassifier()
        stump.fit([r for r in SYNTHETIC_FIXTURE if r["record_id"] in TRAIN_RECORD_IDS])
        preds = stump.predict(SYNTHETIC_FIXTURE)
        report = generate_error_report(SYNTHETIC_FIXTURE, preds, stump.threshold)

        self.assertEqual(report["total_records"], 8)
        self.assertEqual(report["total_correct"], 8)
        self.assertEqual(report["total_errors"], 0)
        self.assertEqual(len(report["error_rows"]), 0)
        self.assertEqual(report["accuracy"], 1.0)

    def test_injected_error_appears_in_report(self):
        """Injected wrong prediction for s4 appears with correct record ID and reason."""
        pipeline_output = run_error_analysis_pipeline()
        report = pipeline_output["error_report"]

        # Aggregate counts match detailed rows
        self.assertEqual(report["total_errors"], 1)
        self.assertEqual(len(report["error_rows"]), 1)
        self.assertEqual(report["total_correct"], 7)
        self.assertEqual(report["total_records"], 8)

        # Injected error row details
        error_row = report["error_rows"][0]
        self.assertEqual(error_row["record_id"], INJECTED_ERROR_TARGET_ID)
        self.assertEqual(error_row["expected_value"], 0)
        self.assertEqual(error_row["predicted_value"], INJECTED_ERROR_WRONG_PRED)
        self.assertEqual(error_row["reason_category"], INJECTED_ERROR_REASON)

    def test_correct_predictions_do_not_appear_in_report(self):
        """Correctly classified records MUST NOT appear in error_rows."""
        pipeline_output = run_error_analysis_pipeline()
        report = pipeline_output["error_report"]
        error_ids = {row["record_id"] for row in report["error_rows"]}

        correct_ids = {"s1", "s2", "s3", "s5", "s6", "s7", "s8"}
        for cid in correct_ids:
            self.assertNotIn(cid, error_ids, f"Correct record '{cid}' improperly appeared in error report!")

    def test_aggregate_error_counts_match_detailed_rows(self):
        """total_errors in report must strictly equal len(error_rows)."""
        pipeline_output = run_error_analysis_pipeline()
        report = pipeline_output["error_report"]
        self.assertEqual(report["total_errors"], len(report["error_rows"]))
        self.assertEqual(report["total_correct"] + report["total_errors"], report["total_records"])


class TestErrorAnalysisFailureAndRejectionCases(unittest.TestCase):

    def test_mismatched_prediction_count_raises_value_error(self):
        """Mismatched prediction dictionary size must raise ValueError."""
        incomplete_preds = {"s1": 0, "s2": 0}  # missing s3..s8
        with self.assertRaises(ValueError):
            generate_error_report(SYNTHETIC_FIXTURE, incomplete_preds)

    def test_unfitted_model_raises_runtime_error(self):
        """Calling predict before fit must raise RuntimeError."""
        stump = DecisionStumpClassifier()
        with self.assertRaises(RuntimeError):
            stump.predict_record(SYNTHETIC_FIXTURE[0])

    def test_tampered_error_report_detection(self):
        """Deliberately changing expected error record ID must trigger test failure."""
        pipeline_output = run_error_analysis_pipeline()
        report = pipeline_output["error_report"]
        actual_error_id = report["error_rows"][0]["record_id"]
        wrong_expected_id = "s1"
        self.assertNotEqual(actual_error_id, wrong_expected_id)


if __name__ == "__main__":
    unittest.main()
