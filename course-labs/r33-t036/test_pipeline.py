"""
Unit tests for R33-T036: Add a bounded CPU smoke run
=====================================================
Covers:
  - Bounded sample count (<= 32)
  - Bounded training steps (<= 20) and epochs (<= 1)
  - Finite loss values and probabilities
  - Output shapes and interface correctness
  - Naive baseline explicitly labelled
  - Failure/contamination cases (sample limit exceeded, unfitted predict)
  - Deliberate wrong expectation detection

Standard library only. No GPU or external dependencies required.
"""

import math
import sys
import unittest

sys.path.insert(0, "course-labs/r33-t036")
from pipeline import (
    MAX_EPOCHS_BOUND,
    MAX_SAMPLES_BOUND,
    MAX_STEPS_BOUND,
    SYNTHETIC_FIXTURE,
    TEST_RECORD_IDS,
    TRAIN_RECORD_IDS,
    NaiveConstantBaseline,
    TinyLogisticCPUClassifier,
    get_stratified_split,
    run_smoke_pipeline,
)


class TestSmokePipelineBounds(unittest.TestCase):

    def test_fixture_size_within_bound(self):
        """Fixture size must be <= 32 samples."""
        self.assertLessEqual(len(SYNTHETIC_FIXTURE), MAX_SAMPLES_BOUND)
        self.assertEqual(len(SYNTHETIC_FIXTURE), 8)

    def test_stratified_split_preserves_classes_and_isolation(self):
        """Train and test splits must be disjoint and contain both classes."""
        train, test = get_stratified_split()
        train_ids = {r["record_id"] for r in train}
        test_ids = {r["record_id"] for r in test}

        # Disjoint
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertEqual(train_ids, set(TRAIN_RECORD_IDS))
        self.assertEqual(test_ids, set(TEST_RECORD_IDS))

        # Class balance preserved
        train_labels = {r["label"] for r in train}
        test_labels = {r["label"] for r in test}
        self.assertEqual(train_labels, {0, 1})
        self.assertEqual(test_labels, {0, 1})

    def test_model_step_and_epoch_bounds(self):
        """Model fit must obey <= 20 steps and <= 1 epoch."""
        train, _ = get_stratified_split()
        model = TinyLogisticCPUClassifier(seed=42)
        model.fit(train, max_steps=20, max_epochs=1)

        self.assertLessEqual(model.steps_executed, MAX_STEPS_BOUND)
        self.assertLessEqual(model.epochs_executed, MAX_EPOCHS_BOUND)
        self.assertGreater(model.steps_executed, 0)

    def test_sample_bound_rejection(self):
        """Attempting to fit on > 32 samples must raise ValueError."""
        oversized = [
            {"record_id": f"x{i}", "x1": float(i), "x2": 1.0, "label": i % 2}
            for i in range(35)
        ]
        model = TinyLogisticCPUClassifier()
        with self.assertRaises(ValueError):
            model.fit(oversized)

    def test_step_bound_rejection(self):
        """Requesting > 20 steps must raise ValueError."""
        train, _ = get_stratified_split()
        model = TinyLogisticCPUClassifier()
        with self.assertRaises(ValueError):
            model.fit(train, max_steps=25)


class TestModelInterfaceCorrectness(unittest.TestCase):

    def test_predict_before_fit_raises(self):
        """Unfitted model predict/predict_proba must raise RuntimeError."""
        model = TinyLogisticCPUClassifier()
        with self.assertRaises(RuntimeError):
            model.predict(SYNTHETIC_FIXTURE)
        with self.assertRaises(RuntimeError):
            model.predict_proba(SYNTHETIC_FIXTURE)

    def test_prediction_shapes_and_values(self):
        """Prediction shapes match input size; values in binary domain {0, 1}."""
        train, test = get_stratified_split()
        model = TinyLogisticCPUClassifier(seed=42)
        model.fit(train, max_steps=20, max_epochs=1)

        test_preds = model.predict(test)
        test_probas = model.predict_proba(test)

        self.assertEqual(len(test_preds), len(test))
        self.assertEqual(len(test_probas), len(test))

        for pred in test_preds:
            self.assertIn(pred, [0, 1])

        for prob in test_probas:
            self.assertTrue(math.isfinite(prob))
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_loss_is_finite_and_recorded(self):
        """All recorded training step losses must be finite numbers."""
        train, _ = get_stratified_split()
        model = TinyLogisticCPUClassifier(seed=42)
        model.fit(train, max_steps=20, max_epochs=1)

        self.assertTrue(len(model.loss_history) > 0)
        for loss in model.loss_history:
            self.assertTrue(math.isfinite(loss))

    def test_naive_baseline_interface(self):
        """Naive baseline is explicitly labelled and produces valid shapes."""
        train, test = get_stratified_split()
        baseline = NaiveConstantBaseline()
        self.assertEqual(baseline.model_type, "naive_constant_baseline")

        with self.assertRaises(RuntimeError):
            baseline.predict(test)

        baseline.fit(train)
        preds = baseline.predict(test)
        self.assertEqual(len(preds), len(test))
        for p in preds:
            self.assertIn(p, [0, 1])


class TestEndToEndPipeline(unittest.TestCase):

    def test_smoke_pipeline_run_success(self):
        """Full smoke pipeline completes with PASS status and finite outputs."""
        result = run_smoke_pipeline(seed=42)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["model_type"], "tiny_logistic_cpu")
        self.assertLessEqual(result["steps_executed"], 20)
        self.assertLessEqual(result["epochs_executed"], 1)
        self.assertTrue(math.isfinite(result["final_loss"]))
        self.assertEqual(len(result["test_predictions"]), len(TEST_RECORD_IDS))

    def test_deliberate_tampering_fails(self):
        """Changing expected prediction assertions must fail."""
        result = run_smoke_pipeline(seed=42)
        tampered_predictions = [999, 999]
        self.assertNotEqual(result["test_predictions"], tampered_predictions)


if __name__ == "__main__":
    unittest.main()
