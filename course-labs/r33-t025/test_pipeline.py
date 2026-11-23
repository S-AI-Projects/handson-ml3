import unittest
import json
import os
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

ErrorAnalysisPipeline = pipeline.ErrorAnalysisPipeline

class TestErrorAnalysis(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)
        self.pipeline = ErrorAnalysisPipeline()

    def test_fixture_integrity(self):
        """Six points with stable IDs, expected clusters are 0,0,0,1,1,1."""
        self.assertEqual(len(self.fixture), 6)
        expected = [r["expected_cluster"] for r in self.fixture]
        self.assertEqual(expected, [0, 0, 0, 1, 1, 1])

    def test_clean_prediction_no_errors(self):
        """Standard predictions match expected exactly with 0 errors in report."""
        preds = self.pipeline.predict_assignments(self.fixture)
        self.assertEqual(preds, [0, 0, 0, 1, 1, 1])
        report = self.pipeline.generate_error_report(self.fixture, preds)
        self.assertEqual(report["total_errors"], 0)
        self.assertEqual(len(report["error_details"]), 0)

    def test_injected_error_report(self):
        """Injected deliberately wrong prediction appears in report; correct do not."""
        preds = self.pipeline.predict_assignments(self.fixture)
        # Deliberately invert prediction for pt_003 (expected 0, predicted 1)
        injected_preds = list(preds)
        injected_preds[2] = 1

        report = self.pipeline.generate_error_report(self.fixture, injected_preds)
        self.assertEqual(report["total_errors"], 1)
        self.assertEqual(len(report["error_details"]), 1)
        
        err = report["error_details"][0]
        self.assertEqual(err["record_id"], "pt_003")
        self.assertEqual(err["expected"], 0)
        self.assertEqual(err["predicted"], 1)
        self.assertEqual(err["reason_category"], "centroid_boundary_misclassification")
        self.assertEqual(report["total_errors"], len(report["error_details"]))

    def test_deliberate_mutation_failure(self):
        """Mutating injected error expectation fails."""
        preds = [0, 0, 1, 1, 1, 1]  # pt_003 error
        report = self.pipeline.generate_error_report(self.fixture, preds)
        mutated_error_id = "pt_001"
        self.assertNotEqual(report["error_details"][0]["record_id"], mutated_error_id)
