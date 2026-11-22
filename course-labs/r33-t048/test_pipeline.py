"""
Unit tests for Course Lab R33-T048 Regression Error Analysis.
"""
import unittest
import json
import os
from pipeline import RegressionErrorAnalyzer

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixture.json")

class TestRegressionErrorAnalysis(unittest.TestCase):
    def setUp(self):
        with open(FIXTURE_PATH, "r") as f:
            self.fixture = json.load(f)
        self.analyzer = RegressionErrorAnalyzer()

    def test_fixture_formula_consistency(self):
        self.assertEqual(len(self.fixture), 6)
        for r in self.fixture:
            x1 = r["x1"]
            expected_target = 2 * x1 + 1
            self.assertEqual(r["target"], expected_target)

    def test_injected_error_appears_in_report(self):
        report = self.analyzer.generate_report(self.fixture, injected_error_id="r6", injected_value=99.0)
        outliers = [row for row in report["detailed_rows"] if row["reason_category"] == "OUTLIER_ANOMALY"]
        self.assertEqual(len(outliers), 1)
        self.assertEqual(outliers[0]["record_id"], "r6")
        self.assertEqual(outliers[0]["predicted"], 99.0)
        self.assertEqual(outliers[0]["expected"], 11.0)

    def test_correct_predictions_not_flagged(self):
        report = self.analyzer.generate_report(self.fixture, injected_error_id="r6")
        for row in report["detailed_rows"]:
            if row["record_id"] != "r6":
                self.assertEqual(row["reason_category"], "EXACT_MATCH")
                self.assertEqual(row["residual"], 0.0)

    def test_aggregate_counts_match_detailed_rows(self):
        report = self.analyzer.generate_report(self.fixture, injected_error_id="r6")
        counts = report["summary_counts"]
        total = sum(counts.values())
        self.assertEqual(total, len(self.fixture))
        self.assertEqual(counts["EXACT_MATCH"], 5)
        self.assertEqual(counts["OUTLIER_ANOMALY"], 1)

    def test_split_design(self):
        train_ids = [r["record_id"] for r in self.fixture if r["split"] == "train"]
        val_ids = [r["record_id"] for r in self.fixture if r["split"] == "validation"]
        test_ids = [r["record_id"] for r in self.fixture if r["split"] == "test"]
        self.assertEqual(train_ids, ["r1", "r2", "r3", "r4"])
        self.assertEqual(val_ids, ["r5"])
        self.assertEqual(test_ids, ["r6"])

if __name__ == "__main__":
    unittest.main()
