"""
Unit tests for R33-T009: Input Perturbation Robustness for Gradient Boosted Classifier
"""

import json
import os
import unittest
from pipeline import (
    load_dataset,
    GradientBoostedStumpClassifier,
    permute_record_order,
    test_robustness_properties,
    IncompatibleShapeError,
    InvalidFeatureValueError,
    DatasetRecord,
)


class TestR33T009InputPerturbationRobustness(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_json = load_dataset(os.path.join(self.base_dir, "fixture.json"), format="json")
        self.records_csv = load_dataset(os.path.join(self.base_dir, "fixture.csv"), format="csv")

        # Train model on training split
        self.train_records = [r for r in self.records_json if r.split == "train"]
        self.val_records = [r for r in self.records_json if r.split == "validation"]
        self.test_records = [r for r in self.records_json if r.split == "test"]

        self.model = GradientBoostedStumpClassifier(n_estimators=3, learning_rate=0.5)
        self.model.fit(self.train_records)

    def test_fixture_structure_and_stratification(self):
        """
        Fixture must contain 8 synthetic rows with x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1].
        Stratified splits: train (4 rows), val (2 rows), test (2 rows) with 50/50 balance.
        """
        self.assertEqual(len(self.records_json), 8)
        for i, r in enumerate(self.records_json):
            self.assertEqual(r.x1, float(i))
            self.assertEqual(r.x2, 1.0)
            expected_label = 0 if i < 4 else 1
            self.assertEqual(r.label, expected_label)
            self.assertEqual(r.group_id, "g0" if expected_label == 0 else "g1")

        # Stratified balance check
        for split_name, split_records in [
            ("train", self.train_records),
            ("validation", self.val_records),
            ("test", self.test_records),
        ]:
            pos = sum(1 for r in split_records if r.label == 1)
            neg = sum(1 for r in split_records if r.label == 0)
            self.assertEqual(pos, neg, f"Split '{split_name}' is not 50/50 balanced.")

    def test_csv_and_json_parity(self):
        """CSV and JSON fixtures must load identical data."""
        self.assertEqual(len(self.records_json), len(self.records_csv))
        for r_j, r_c in zip(self.records_json, self.records_csv):
            self.assertEqual(r_j.record_id, r_c.record_id)
            self.assertEqual(r_j.group_id, r_c.group_id)
            self.assertEqual(r_j.x1, r_c.x1)
            self.assertEqual(r_j.x2, r_c.x2)
            self.assertEqual(r_j.label, r_c.label)
            self.assertEqual(r_j.split, r_c.split)

    def test_gradient_boosted_classifier_accuracy(self):
        """Model must achieve 100% accuracy on train, validation, and test splits."""
        for split_name, split_records in [
            ("train", self.train_records),
            ("validation", self.val_records),
            ("test", self.test_records),
        ]:
            preds = self.model.predict_batch(split_records)
            acc = sum(1 for r in split_records if preds[r.record_id] == r.label) / len(split_records)
            self.assertEqual(acc, 1.0, f"Imperfection on split '{split_name}'")

    def test_record_order_permutation_invariance_with_stable_ids(self):
        """
        Permuting record sequence within a batch must preserve identical predictions per record ID.
        """
        # Test on the test split
        preds_orig = self.model.predict_batch(self.test_records)
        probas_orig = self.model.predict_proba_batch(self.test_records)

        # Reverse order
        permuted_test = list(reversed(self.test_records))
        preds_perm = self.model.predict_batch(permuted_test)
        probas_perm = self.model.predict_proba_batch(permuted_test)

        for r in self.test_records:
            self.assertEqual(
                preds_orig[r.record_id],
                preds_perm[r.record_id],
                f"Prediction mismatch for record {r.record_id} under order permutation",
            )
            self.assertAlmostEqual(
                probas_orig[r.record_id],
                probas_perm[r.record_id],
                places=9,
                msg=f"Probability mismatch for record {r.record_id} under order permutation",
            )

        # Test on all records with various random seeds
        for seed in [1, 42, 999]:
            shuffled = permute_record_order(self.records_json, seed=seed)
            shuffled_preds = self.model.predict_batch(shuffled)
            all_orig_preds = self.model.predict_batch(self.records_json)
            for r in self.records_json:
                self.assertEqual(all_orig_preds[r.record_id], shuffled_preds[r.record_id])

    def test_unsupported_shape_rejected(self):
        """Unsupported feature counts (e.g. 1 or 3 features) must raise IncompatibleShapeError."""
        with self.assertRaises(IncompatibleShapeError):
            self.model.validate_features([1.0])

        with self.assertRaises(IncompatibleShapeError):
            self.model.validate_features([1.0, 2.0, 3.0])

    def test_unsupported_invalid_value_rejected(self):
        """Invalid feature values (NaN or Inf) must raise InvalidFeatureValueError."""
        with self.assertRaises(InvalidFeatureValueError):
            self.model.validate_features([float("nan"), 1.0])

        with self.assertRaises(InvalidFeatureValueError):
            self.model.validate_features([1.0, float("inf")])

    def test_robustness_verification_result_invariants(self):
        """Verify the structured robustness verification result."""
        result = test_robustness_properties(self.model, self.records_json, permutation_seed=42)

        self.assertTrue(result.order_invariance_verified)
        self.assertTrue(result.unsupported_shape_rejected)
        self.assertTrue(result.unsupported_value_rejected)
        self.assertEqual(result.accuracy_original, 1.0)
        self.assertEqual(result.accuracy_permuted, 1.0)
        self.assertEqual(result.num_records_tested, 8)
        self.assertFalse(result.claims_broad_adversarial_robustness)

    def test_deliberate_mutation_fails(self):
        """Mutation test: changing expected prediction label causes assertion failure."""
        preds = self.model.predict_batch(self.test_records)
        mutated_wrong_label = 999
        self.assertNotEqual(preds[self.test_records[0].record_id], mutated_wrong_label)


if __name__ == "__main__":
    unittest.main()
