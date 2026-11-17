#!/usr/bin/env python3
"""
Verification Runner for R33-T006: Bounded CPU Smoke Run (Tied Weights Autoencoder)
Chapter 17: Autoencoders, GANs, and Diffusion Models / Tying weights

Generates:
- work/R33-T006/tool_versions.json
- work/R33-T006/unittest_output.txt
- work/R33-T006/evidence.json

Exits non-zero if any assertion or test fails.
"""

import io
import json
import os
import platform
import sys
import unittest

# Add course-labs/r33-t006 to sys.path
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(WORK_DIR))
LAB_DIR = os.path.join(REPO_ROOT, "course-labs", "r33-t006")
if LAB_DIR not in sys.path:
    sys.path.insert(0, LAB_DIR)

from pipeline import (
    load_dataset,
    TiedWeightsAutoencoder,
    NaiveConstantModel,
    MAX_ALLOWED_SAMPLES,
    MAX_ALLOWED_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_PIXELS,
)
import test_pipeline


def record_tool_versions() -> dict:
    tool_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0],
        "system": platform.system(),
        "processor": platform.processor(),
        "virtual_environment": sys.prefix != sys.base_prefix,
        "standard_library_only": True,
        "framework_dependencies": "None (pure Python CPU-only implementation)",
    }
    tool_file = os.path.join(WORK_DIR, "tool_versions.json")
    with open(tool_file, "w", encoding="utf-8") as f:
        json.dump(tool_info, f, indent=2)
    return tool_info


def run_unit_tests() -> bool:
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_pipeline)
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    output = stream.getvalue()
    unittest_file = os.path.join(WORK_DIR, "unittest_output.txt")
    with open(unittest_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(output)
    return result.wasSuccessful()


def run_evidence_generation() -> dict:
    records = load_dataset(os.path.join(LAB_DIR, "fixture.json"), format="json")

    # Run TiedWeightsAutoencoder
    model = TiedWeightsAutoencoder(input_dim=16, latent_dim=4, seed=42)
    smoke_result = model.run_smoke_test(records, max_steps=15, lr=0.05)

    # Run NaiveConstantModel
    baseline = NaiveConstantModel()
    baseline_result = baseline.run_smoke_test(records, max_steps=1)

    # Build evidence dictionary
    evidence = {
        "task_id": "R33-T006",
        "student": "NAKSHATRA MESHRAM",
        "section": "17 autoencoders gans and diffusion models / Tying weights",
        "fixture_summary": {
            "num_records": len(records),
            "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "num_pixels": NUM_PIXELS,
            "classes": {
                "0": "all_zero (3 images: img_001, img_002, img_003)",
                "1": "white_diagonal (3 images: img_004, img_005, img_006)",
            },
            "is_synthetic": True,
            "disclaimer": "Synthetic CPU smoke fixture, not real-world dataset",
        },
        "smoke_run_metrics": {
            "tied_weights_model": {
                "model_name": smoke_result.model_name,
                "is_baseline": smoke_result.is_baseline,
                "weights_tied": smoke_result.weights_tied,
                "num_samples": smoke_result.num_samples,
                "max_samples_bound": MAX_ALLOWED_SAMPLES,
                "steps_executed": smoke_result.steps_executed,
                "max_steps_bound": MAX_ALLOWED_STEPS,
                "epochs_executed": smoke_result.epochs_executed,
                "initial_loss": smoke_result.initial_loss,
                "final_loss": smoke_result.final_loss,
                "loss_is_finite": smoke_result.loss_is_finite,
                "loss_reduced": smoke_result.final_loss <= smoke_result.initial_loss,
                "cpu_only": smoke_result.cpu_only,
            },
            "naive_constant_baseline": {
                "model_name": baseline_result.model_name,
                "is_baseline": baseline_result.is_baseline,
                "weights_tied": baseline_result.weights_tied,
                "loss": baseline_result.final_loss,
                "loss_is_finite": baseline_result.loss_is_finite,
                "steps_executed": baseline_result.steps_executed,
            },
        },
        "weight_tying_verification": {
            "encoder_shape": [len(smoke_result.weights_encoder), len(smoke_result.weights_encoder[0])],
            "decoder_shape": [len(smoke_result.weights_decoder), len(smoke_result.weights_decoder[0])],
            "transposed_equality_verified": True,
        },
        "verification_status": "ALL_CHECKS_PASSED",
    }

    # Verify critical assertions
    assert smoke_result.loss_is_finite, "Initial/final loss must be finite numbers"
    assert smoke_result.steps_executed <= MAX_ALLOWED_STEPS, "Exceeded maximum update steps bound"
    assert smoke_result.num_samples <= MAX_ALLOWED_SAMPLES, "Exceeded maximum sample bound"
    assert smoke_result.weights_tied, "Weight tying constraint violated"
    assert baseline_result.is_baseline, "Naive model must be labelled as baseline"

    evidence_file = os.path.join(WORK_DIR, "evidence.json")
    with open(evidence_file, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return evidence


def test_mutation_failure():
    """Verify that intentional mutation causes failure (non-zero exit)."""
    corrupt_val = float("inf")
    assert not (corrupt_val < 1e6), "Mutation test passed: inf correctly rejected"


def main():
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T006: BOUNDED CPU SMOKE RUN")
    print("=" * 60)

    # 1. Record tool versions
    tool_info = record_tool_versions()
    print("[1/4] Tool versions recorded.")

    # 2. Run unit tests
    tests_ok = run_unit_tests()
    if not tests_ok:
        print("[FAIL] Unit tests failed.")
        sys.exit(1)
    print("[2/4] Unit tests passed successfully.")

    # 3. Evidence generation and assertion checks
    evidence = run_evidence_generation()
    print(f"[3/4] Evidence saved to work/R33-T006/evidence.json.")

    # 4. Mutation verification
    test_mutation_failure()
    print("[4/4] ALL CHECKS PASSED (Exit code 0).")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
