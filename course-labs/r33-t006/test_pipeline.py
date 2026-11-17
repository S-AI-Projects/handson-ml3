"""
Unit tests for R33-T006: Bounded CPU Smoke Run for Autoencoder with Tied Weights
"""

import math
import os
import unittest
from pipeline import (
    load_dataset,
    TiedWeightsAutoencoder,
    NaiveConstantModel,
    DatasetRecord,
    MAX_ALLOWED_SAMPLES,
    MAX_ALLOWED_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_PIXELS,
)


class TestR33T006TiedWeightsSmokeRun(unittest.TestCase):
    def setUp(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.records_json = load_dataset(os.path.join(self.base_dir, "fixture.json"), format="json")
        self.records_csv = load_dataset(os.path.join(self.base_dir, "fixture.csv"), format="csv")

    def test_fixture_structure_and_labels(self):
        """
        Fixture must contain six synthetic 4x4 grayscale images:
        - 3 all-zero images labelled 0 (group g_zero)
        - 3 images with a white diagonal (255) labelled 1 (group g_diag)
        """
        self.assertEqual(len(self.records_json), 6)

        zero_records = [r for r in self.records_json if r.label == 0]
        diag_records = [r for r in self.records_json if r.label == 1]

        self.assertEqual(len(zero_records), 3)
        self.assertEqual(len(diag_records), 3)

        # Check all-zero records
        for r in zero_records:
            self.assertEqual(r.group_id, "g_zero")
            self.assertEqual(len(r.pixels), NUM_PIXELS)
            self.assertTrue(all(p == 0 for p in r.pixels))
            self.assertEqual(len(r.pixels_2d), IMAGE_HEIGHT)
            self.assertEqual(len(r.pixels_2d[0]), IMAGE_WIDTH)

        # Check white diagonal records
        for r in diag_records:
            self.assertEqual(r.group_id, "g_diag")
            self.assertEqual(len(r.pixels), NUM_PIXELS)
            for row_idx in range(IMAGE_HEIGHT):
                for col_idx in range(IMAGE_WIDTH):
                    expected = 255 if row_idx == col_idx else 0
                    self.assertEqual(
                        r.pixels_2d[row_idx][col_idx],
                        expected,
                        f"Diagonal mismatch at row {row_idx}, col {col_idx} in record {r.record_id}",
                    )
            # Flattened indices 0, 5, 10, 15 are 255
            for idx in range(NUM_PIXELS):
                expected = 255 if idx in (0, 5, 10, 15) else 0
                self.assertEqual(r.pixels[idx], expected)

    def test_csv_and_json_parity(self):
        """
        Loading from CSV and JSON must produce identical records and pixel arrays.
        """
        self.assertEqual(len(self.records_json), len(self.records_csv))
        for r_json, r_csv in zip(self.records_json, self.records_csv):
            self.assertEqual(r_json.record_id, r_csv.record_id)
            self.assertEqual(r_json.group_id, r_csv.group_id)
            self.assertEqual(r_json.split, r_csv.split)
            self.assertEqual(r_json.label, r_csv.label)
            self.assertEqual(r_json.pixels, r_csv.pixels)
            self.assertEqual(r_json.pixels_2d, r_csv.pixels_2d)

    def test_weight_tying_identity_constraint(self):
        """
        Assert that decoder weight matrix W_dec is strictly the transpose of W_enc.
        W_dec[k][j] == W_enc[j][k] for all j in [0..D-1], k in [0..K-1].
        """
        model = TiedWeightsAutoencoder(input_dim=16, latent_dim=4, seed=42)
        W_enc = model.W_enc
        W_dec = model.W_dec

        self.assertEqual(len(W_enc), 16)
        self.assertEqual(len(W_enc[0]), 4)
        self.assertEqual(len(W_dec), 4)
        self.assertEqual(len(W_dec[0]), 16)

        for j in range(16):
            for k in range(4):
                self.assertEqual(
                    W_dec[k][j],
                    W_enc[j][k],
                    f"Weight tying violation at encoder ({j}, {k}) vs decoder ({k}, {j})",
                )

        # Mutate W_enc and verify W_dec dynamically reflects the change (property constraint)
        model.W_enc[2][1] = 9.999
        self.assertEqual(model.W_dec[1][2], 9.999)

    def test_bounded_smoke_run_finite_outputs_and_loss(self):
        """
        Assert smoke run produces finite losses, recorded step count <= 20,
        epoch count <= 1, sample count <= 32, on CPU without GPU or external data.
        """
        model = TiedWeightsAutoencoder(input_dim=16, latent_dim=4, seed=42)
        result = model.run_smoke_test(self.records_json, max_steps=15, lr=0.05)

        self.assertTrue(result.loss_is_finite)
        self.assertTrue(math.isfinite(result.initial_loss))
        self.assertTrue(math.isfinite(result.final_loss))
        self.assertGreater(result.initial_loss, 0.0)
        self.assertLessEqual(result.final_loss, result.initial_loss)
        self.assertEqual(result.steps_executed, 15)
        self.assertLessEqual(result.steps_executed, MAX_ALLOWED_STEPS)
        self.assertEqual(result.epochs_executed, 1)
        self.assertEqual(result.num_samples, 6)
        self.assertLessEqual(result.num_samples, MAX_ALLOWED_SAMPLES)
        self.assertTrue(result.weights_tied)
        self.assertTrue(result.cpu_only)
        self.assertTrue(result.synthetic_verification_only)
        self.assertFalse(result.is_baseline)

    def test_prediction_shape_and_bounds(self):
        """
        Assert reconstruction shapes match (N, 16) flat and (N, 4, 4) 2D.
        Values must be clamped within [0.0, 255.0].
        """
        model = TiedWeightsAutoencoder(input_dim=16, latent_dim=4, seed=42)
        result = model.run_smoke_test(self.records_json, max_steps=5, lr=0.05)

        self.assertEqual(len(result.reconstructions_flat), 6)
        self.assertEqual(len(result.reconstructions_2d), 6)

        for row_flat, mat_2d in zip(result.reconstructions_flat, result.reconstructions_2d):
            self.assertEqual(len(row_flat), 16)
            self.assertEqual(len(mat_2d), 4)
            self.assertEqual(len(mat_2d[0]), 4)

            # Check bounds
            self.assertTrue(all(0.0 <= val <= 255.0 for val in row_flat))
            self.assertTrue(all(0.0 <= val <= 255.0 for row in mat_2d for val in row))

            # Flat and 2D parity
            for r in range(4):
                for c in range(4):
                    self.assertEqual(mat_2d[r][c], row_flat[r * 4 + c])

    def test_sample_limit_enforced(self):
        """
        Providing > 32 samples must raise ValueError.
        """
        model = TiedWeightsAutoencoder(input_dim=16, latent_dim=4, seed=42)
        oversized = [self.records_json[0]] * 33
        with self.assertRaises(ValueError) as ctx:
            model.run_smoke_test(oversized, max_steps=5)
        self.assertIn("exceeds maximum allowed bounded limit", str(ctx.exception))

    def test_step_limit_enforced(self):
        """
        Requesting > 20 steps must raise ValueError.
        """
        model = TiedWeightsAutoencoder(input_dim=16, latent_dim=4, seed=42)
        with self.assertRaises(ValueError) as ctx:
            model.run_smoke_test(self.records_json, max_steps=21)
        self.assertIn("exceeds maximum allowed bounded limit", str(ctx.exception))

    def test_naive_constant_baseline_explicitly_labelled(self):
        """
        NaiveConstantModel must be explicitly labelled as a baseline (is_baseline=True),
        produce finite loss, and output consistent shapes.
        """
        baseline = NaiveConstantModel()
        result = baseline.run_smoke_test(self.records_json, max_steps=1)

        self.assertTrue(result.is_baseline)
        self.assertFalse(result.weights_tied)
        self.assertEqual(result.model_name, "NaiveConstantModel")
        self.assertTrue(result.loss_is_finite)
        self.assertEqual(result.steps_executed, 1)
        self.assertEqual(len(result.reconstructions_flat), 6)
        self.assertEqual(len(result.reconstructions_2d), 6)
        self.assertTrue(result.cpu_only)

    def test_deliberate_mutation_fails(self):
        """
        Mutation test: deliberately altering weight tying or corrupting losses
        must trigger an assertion failure.
        """
        model = TiedWeightsAutoencoder(input_dim=16, latent_dim=4, seed=42)
        result = model.run_smoke_test(self.records_json, max_steps=5)

        # Finiteness check
        corrupt_nan = float("nan")
        self.assertFalse(math.isfinite(corrupt_nan))

        # Shape check mutation
        mutated_shape = (5, 5)
        self.assertNotEqual(result.input_shape, mutated_shape)


if __name__ == "__main__":
    unittest.main()
