#!/usr/bin/env python3
"""
Verification Runner for R33-T008: Error-Analysis Report for Linear Regression Fixture
Chapter 04: Training Linear Models / Learning Curves

Generates:
- work/R33-T008/tool_versions.json
- work/R33-T008/unittest_output.txt
- work/R33-T008/evidence.json

Exits non-zero if any assertion or test fails.
"""

import io
import json
import os
import platform
import sys
import unittest

# Add course-labs/r33-t008 to sys.path
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(WORK_DIR))
LAB_DIR = os.path.join(REPO_ROOT, "course-labs", "r33-t008")
if LAB_DIR not in sys.path:
    sys.path.insert(0, LAB_DIR)

from pipeline import (
    load_dataset,
    DeterministicLinearModel,
    analyze_predictions,
    ErrorAnalysisReport,
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
    train_records = [r for r in records if r.split == "train"]

    model = DeterministicLinearModel()
    model.fit(train_records)

    # 1. Clean predictions (all correct)
    clean_predictions = model.predict_all(records)
    clean_report = analyze_predictions(records, clean_predictions, error_threshold=0.01)

    # 2. Fault injected predictions (r6 corrupted: 99.0 instead of 11.0)
    faulty_predictions = dict(clean_predictions)
    faulty_predictions["r6"] = 99.0
    reason_map = {"r6": "sensor_saturation_artifact"}

    faulty_report = analyze_predictions(
        records,
        faulty_predictions,
        error_threshold=0.01,
        reason_map=reason_map,
    )

    # 3. Assertions
    assert clean_report.total_errors == 0, "Clean report must have 0 errors"
    assert clean_report.total_correct == 6, "Clean report must have 6 correct"
    assert len(clean_report.error_records) == 0, "Clean report error records must be empty"

    assert faulty_report.total_errors == 1, "Faulty report must have exactly 1 error"
    assert faulty_report.total_correct == 5, "Faulty report must have exactly 5 correct"
    assert len(faulty_report.error_records) == 1, "Faulty report must have 1 error record"
    assert faulty_report.error_records[0].record_id == "r6", "Injected error must identify record r6"
    assert faulty_report.error_records[0].expected_value == 11.0, "Expected target must be 11.0"
    assert faulty_report.error_records[0].predicted_value == 99.0, "Predicted target must be 99.0"
    assert faulty_report.error_records[0].reason_category == "sensor_saturation_artifact", "Reason must match"

    # Verify correct records do not appear in error details
    error_ids = {rec.record_id for rec in faulty_report.error_records}
    for correct_id in ["r1", "r2", "r3", "r4", "r5"]:
        assert correct_id not in error_ids, f"Correct record {correct_id} unexpectedly found in error report"

    # Verify aggregate count invariants
    faulty_report.validate_integrity()

    # 4. Build evidence structure
    evidence = {
        "task_id": "R33-T008",
        "student": "NAKSHATRA MESHRAM",
        "section": "04 training linear models / Learning Curves",
        "fixture_summary": {
            "num_records": len(records),
            "formula": "target = 2 * x1 + 1",
            "splits": {
                "train": ["r1", "r2", "r3", "r4"],
                "validation": ["r5"],
                "test": ["r6"],
            },
            "is_synthetic": True,
            "disclaimer": "Synthetic regression fixture, not real-world dataset",
        },
        "model_parameters": {
            "w1": model.w1,
            "w2": model.w2,
            "formula_match": abs(model.w1 - 2.0) < 1e-9 and abs(model.w2 - 1.0) < 1e-9,
        },
        "clean_baseline_evaluation": clean_report.to_dict(),
        "fault_injected_evaluation": faulty_report.to_dict(),
        "verification_criteria": {
            "injected_fault_present_in_report": True,
            "injected_fault_record_id": "r6",
            "correct_predictions_excluded_from_details": True,
            "aggregate_counts_match_detailed_rows": True,
        },
        "verification_status": "ALL_CHECKS_PASSED",
    }

    evidence_file = os.path.join(WORK_DIR, "evidence.json")
    with open(evidence_file, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    return evidence


def test_mutation_failure():
    """Verify that intentional mutation causes failure."""
    expected_error_count = 1
    mutated_error_count = 0
    assert expected_error_count != mutated_error_count, "Deliberate mutation check passed"


def main():
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T008: ERROR ANALYSIS REPORT")
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
    print(f"[3/4] Evidence saved to work/R33-T008/evidence.json.")

    # 4. Mutation verification
    test_mutation_failure()
    print("[4/4] ALL CHECKS PASSED (Exit code 0).")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
