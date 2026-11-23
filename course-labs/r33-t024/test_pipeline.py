import unittest
import json
import os
import importlib.util

pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline", pipeline_path)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

compute_precision_recall = pipeline.compute_precision_recall
compute_mae = pipeline.compute_mae
ImageClassifierMetricEvaluator = pipeline.ImageClassifierMetricEvaluator

class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture.json")
        with open(fixture_path) as f:
            self.fixture = json.load(f)

    def test_fixture_integrity(self):
        """Asserts 6 synthetic 4x4 images: 3 all-zero (label 0) and 3 diagonal 255 (label 1)."""
        self.assertEqual(len(self.fixture), 6)
        zeros = [r for r in self.fixture if r["label"] == 0]
        diags = [r for r in self.fixture if r["label"] == 1]
        self.assertEqual(len(zeros), 3)
        self.assertEqual(len(diags), 3)
        for r in zeros:
            self.assertEqual(r["pixels"], [[0]*4 for _ in range(4)])
        for r in diags:
            expected_diag = [[255 if i == j else 0 for j in range(4)] for i in range(4)]
            self.assertEqual(r["pixels"], expected_diag)

    def test_classification_metric_specification(self):
        """truth [1,1,0,0] and prediction [1,0,1,0] -> precision=0.5, recall=0.5."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        precision, recall = compute_precision_recall(y_true, y_pred)
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.5)

    def test_regression_mae_specification(self):
        """truth [1,2,3] and prediction [1,3,2] -> MAE = 2/3."""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 3.0, 2.0]
        mae = compute_mae(y_true, y_pred)
        self.assertAlmostEqual(mae, 2.0 / 3.0, places=6)

    def test_zero_denominator_behavior(self):
        """Zero division defaults to specified constant without ZeroDivisionError."""
        prec, rec = compute_precision_recall([0, 0], [0, 0], zero_division=0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)

    def test_evaluator_on_fixture(self):
        """Evaluator computes accurate metrics on image fixture."""
        evaluator = ImageClassifierMetricEvaluator()
        # Perfect predictions on fixture
        preds = [r["label"] for r in self.fixture]
        metrics = evaluator.evaluate(self.fixture, preds)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_deliberate_mutation_failure(self):
        """Deliberately mutating precision assertion must fail."""
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 1, 0]
        precision, recall = compute_precision_recall(y_true, y_pred)
        mutated_precision = 0.99
        self.assertNotEqual(precision, mutated_precision)
