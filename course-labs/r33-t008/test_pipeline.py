"""
Unit tests for R33-T008: Error-Analysis Report for Linear Regression Fixture
"""

import json
import os
import unittest
from pipeline import (
    load_dataset,
    DeterministicLinearModel,
    analyze_predictions,
    ErrorAnalysisReport,
    DatasetRecord,
)


class TestR33T008ErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_json = load_dataset(os.path.join(self.base_dir, "fixture.json"), format="json")
        self.records_csv = load_dataset(os.path.join(self.base_dir, "fixture.csv"), format="csv")

        # Train deterministic model on training partition (r1..r4)
        train_records = [r for r in self.records_json if r.split == "train"]
        self.model = DeterministicLinearModel()
        self.model.fit(train_records)

    def test_fixture_records_and_formula(self):
        """
        Fixture must contain six synthetic rows obeying target = 2*x1 + 1:
        x1=[0..5], x2=1, splits: train r1..r4, val r5, test r6.
        """
        self.assertEqual(len(self.records_json), 6)
        expected_splits = {
            "r1": "train",
            "r2": "train",
            "r3": "train",
            "r4": "train",
            "r5": "validation",
            "r6": "test",
        }
        for i, r in enumerate(self.records_json):
            self.assertEqual(r.x1, float(i))
            self.assertEqual(r.x2, 1.0)
            self.assertEqual(r.target, 2.0 * float(i) + 1.0)
            self.assertEqual(r.split, expected_splits[r.record_id])

    def test_csv_and_json_parity(self):
        """CSV and JSON fixtures must load identical data."""
        self.assertEqual(len(self.records_json), len(self.records_csv))
        for r_j, r_c in zip(self.records_json, self.records_csv):
            self.assertEqual(r_j.record_id, r_c.record_id)
            self.assertEqual(r_j.x1, r_c.x1)
            self.assertEqual(r_j.x2, r_c.x2)
            self.assertEqual(r_j.target, r_c.target)
            self.assertEqual(r_j.split, r_c.split)

    def test_deterministic_model_perfect_fit(self):
        """Model fitted on r1..r4 must learn w1=2.0, w2=1.0 and predict accurately."""
        self.assertAlmostEqual(self.model.w1, 2.0, places=9)
        self.assertAlmostEqual(self.model.w2, 1.0, places=9)

        predictions = self.model.predict_all(self.records_json)
        self.assertAlmostEqual(predictions["r1"], 1.0, places=9)
        self.assertAlmostEqual(predictions["r5"], 9.0, places=9)
        self.assertAlmostEqual(predictions["r6"], 11.0, places=9)

    def test_baseline_error_report_zero_errors(self):
        """
        Without fault injection, all predictions are correct.
        Error report must show 0 errors, 6 correct, and an empty error_records list.
        """
        predictions = self.model.predict_all(self.records_json)
        report = analyze_predictions(self.records_json, predictions, error_threshold=0.01)

        self.assertEqual(report.total_evaluated, 6)
        self.assertEqual(report.total_correct, 6)
        self.assertEqual(report.total_errors, 0)
        self.assertEqual(len(report.error_records), 0)
        self.assertEqual(len(report.aggregate_counts_by_reason), 0)
        self.assertAlmostEqual(report.mean_absolute_error, 0.0, places=9)

    def test_injected_wrong_prediction_detected(self):
        """
        Inject deliberately wrong prediction for record r6 (e.g. 99.0 instead of 11.0).
        1. The injected wrong prediction must appear in the report with the correct record ID (r6).
        2. Correct predictions (r1..r5) must NOT appear in the detailed rows.
        3. Reason category must match the specified reason.
        """
        predictions = self.model.predict_all(self.records_json)
        # Deliberate wrong prediction
        predictions["r6"] = 99.0

        reason_map = {"r6": "sensor_saturation_artifact"}
        report = analyze_predictions(
            self.records_json,
            predictions,
            error_threshold=0.01,
            reason_map=reason_map,
        )

        self.assertEqual(report.total_evaluated, 6)
        self.assertEqual(report.total_correct, 5)
        self.assertEqual(report.total_errors, 1)

        # Assert exactly 1 error row in detail
        self.assertEqual(len(report.error_records), 1)
        err = report.error_records[0]

        # Verify correct record ID
        self.assertEqual(err.record_id, "r6")
        self.assertEqual(err.split, "test")
        self.assertEqual(err.x1, 5.0)
        self.assertEqual(err.x2, 1.0)
        self.assertEqual(err.expected_value, 11.0)
        self.assertEqual(err.predicted_value, 99.0)
        self.assertAlmostEqual(err.residual, 88.0, places=9)
        self.assertAlmostEqual(err.absolute_error, 88.0, places=9)
        self.assertEqual(err.reason_category, "sensor_saturation_artifact")

        # Correct predictions must NOT appear
        error_ids = {rec.record_id for rec in report.error_records}
        for correct_id in ["r1", "r2", "r3", "r4", "r5"]:
            self.assertNotIn(correct_id, error_ids)

    def test_aggregate_error_counts_match_detailed_rows(self):
        """
        Aggregate error counts must match the detailed rows exactly:
        - total_errors == len(error_records)
        - sum(aggregate_counts_by_reason.values()) == total_errors
        - total_evaluated == total_correct + total_errors
        """
        predictions = self.model.predict_all(self.records_json)
        predictions["r6"] = 99.0

        report = analyze_predictions(
            self.records_json,
            predictions,
            error_threshold=0.01,
            reason_map={"r6": "sensor_saturation_artifact"},
        )

        self.assertEqual(report.total_errors, len(report.error_records))
        self.assertEqual(
            sum(report.aggregate_counts_by_reason.values()), report.total_errors
        )
        self.assertEqual(
            report.total_evaluated, report.total_correct + report.total_errors
        )
        self.assertEqual(
            report.aggregate_counts_by_reason.get("sensor_saturation_artifact"), 1
        )

    def test_multiple_injected_errors_aggregate_correctly(self):
        """
        Multiple injected errors (e.g. r5 and r6) must be categorized and counted accurately.
        """
        predictions = self.model.predict_all(self.records_json)
        predictions["r5"] = 25.0  # Expected 9.0
        predictions["r6"] = -5.0  # Expected 11.0

        reason_map = {
            "r5": "extrapolation_uncertainty",
            "r6": "sensor_saturation_artifact",
        }
        report = analyze_predictions(
            self.records_json,
            predictions,
            error_threshold=0.01,
            reason_map=reason_map,
        )

        self.assertEqual(report.total_errors, 2)
        self.assertEqual(report.total_correct, 4)
        self.assertEqual(len(report.error_records), 2)
        self.assertEqual(report.aggregate_counts_by_reason["extrapolation_uncertainty"], 1)
        self.assertEqual(report.aggregate_counts_by_reason["sensor_saturation_artifact"], 1)

        err_ids = [rec.record_id for rec in report.error_records]
        self.assertIn("r5", err_ids)
        self.assertIn("r6", err_ids)
        for correct_id in ["r1", "r2", "r3", "r4"]:
            self.assertNotIn(correct_id, err_ids)

    def test_deliberate_mutation_fails(self):
        """Mutation test: changing expected error record ID triggers assertion failure."""
        predictions = self.model.predict_all(self.records_json)
        predictions["r6"] = 99.0
        report = analyze_predictions(
            self.records_json,
            predictions,
            error_threshold=0.01,
            reason_map={"r6": "sensor_saturation_artifact"},
        )
        mutated_wrong_id = "r1"
        self.assertNotEqual(report.error_records[0].record_id, mutated_wrong_id)


if __name__ == "__main__":
    unittest.main()
