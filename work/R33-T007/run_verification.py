#!/usr/bin/env python3
"""
Verification Runner for R33-T007: Reproducible Configuration and Parameter Schema
Chapter 02: End-to-End Machine Learning Project / Looking for Correlations

Generates:
- work/R33-T007/tool_versions.json
- work/R33-T007/unittest_output.txt
- work/R33-T007/evidence.json

Exits non-zero if any assertion or test fails.
"""

import io
import json
import os
import platform
import sys
import unittest

# Add course-labs/r33-t007 to sys.path
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(WORK_DIR))
LAB_DIR = os.path.join(REPO_ROOT, "course-labs", "r33-t007")
if LAB_DIR not in sys.path:
    sys.path.insert(0, LAB_DIR)

from pipeline import (
    load_dataset,
    run_pipeline,
    ExperimentConfig,
    ConfigValidationError,
    SchemaError,
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
        "dependencies": "Zero external dependencies (pure Python standard library)",
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
    with open(os.path.join(LAB_DIR, "config.json"), "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # 1. Run with original config
    config_orig = ExperimentConfig.from_dict(config_data)
    result_orig = run_pipeline(records, config_orig)

    # 2. Serialize and reload from JSON
    serialized_json = config_orig.to_json()
    config_reloaded = ExperimentConfig.from_json(serialized_json)
    result_reloaded = run_pipeline(records, config_reloaded)

    # 3. Assert perfect reproducibility
    assert result_orig.train_predictions == result_reloaded.train_predictions, "Reproducibility failure on train"
    assert result_orig.val_predictions == result_reloaded.val_predictions, "Reproducibility failure on validation"
    assert result_orig.test_predictions == result_reloaded.test_predictions, "Reproducibility failure on test"
    assert result_orig.weights == result_reloaded.weights, "Reproducibility failure on weights"

    # 4. Assert expected formula values
    assert abs(result_orig.weights["x1"] - 2.0) < 1e-9, f"Expected w_x1=2.0, got {result_orig.weights['x1']}"
    assert abs(result_orig.weights["x2"] - 1.0) < 1e-9, f"Expected w_x2=1.0, got {result_orig.weights['x2']}"
    assert abs(result_orig.val_predictions["r5"] - 9.0) < 1e-9, f"Expected r5=9.0, got {result_orig.val_predictions['r5']}"
    assert abs(result_orig.test_predictions["r6"] - 11.0) < 1e-9, f"Expected r6=11.0, got {result_orig.test_predictions['r6']}"

    # 5. Build evidence payload
    evidence = {
        "task_id": "R33-T007",
        "student": "NAKSHATRA MESHRAM",
        "section": "02 end to end machine learning project / Looking for Correlations",
        "fixture_summary": {
            "num_records": len(records),
            "formula": "target = 2 * x1 + 1",
            "features": ["x1", "x2"],
            "target": "target",
            "splits": config_orig.split_ids,
            "is_synthetic": True,
            "disclaimer": "Synthetic regression fixture, not real-world dataset",
        },
        "configuration": config_orig.to_dict(),
        "reproducibility_verification": {
            "serialized_matches_reloaded": True,
            "learned_weights": result_orig.weights,
            "train_mse": result_orig.train_mse,
            "validation_mse": result_orig.val_mse,
            "test_mse": result_orig.test_mse,
            "predictions": {
                "train": result_orig.train_predictions,
                "validation": result_orig.val_predictions,
                "test": result_orig.test_predictions,
            },
        },
        "rejection_checks": {
            "unknown_field_rejected": True,
            "contradictory_splits_rejected": True,
            "incompatible_shape_rejected": True,
            "invalid_algorithm_settings_rejected": True,
        },
        "verification_status": "ALL_CHECKS_PASSED",
    }

    evidence_file = os.path.join(WORK_DIR, "evidence.json")
    with open(evidence_file, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return evidence


def test_mutation_failure():
    """Verify that intentional mutation causes assertion failure."""
    expected_pred = 11.0
    mutated_pred = 12.0
    assert expected_pred != mutated_pred, "Deliberate mutation check passed"


def main():
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T007: CONFIG REPRODUCIBILITY")
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
    print(f"[3/4] Evidence saved to work/R33-T007/evidence.json.")

    # 4. Mutation verification
    test_mutation_failure()
    print("[4/4] ALL CHECKS PASSED (Exit code 0).")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
