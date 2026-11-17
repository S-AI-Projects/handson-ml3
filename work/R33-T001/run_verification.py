"""
Verification Runner for R33-T001.
Saves inputs, expected vs actual results, and tool versions into work/R33-T001/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

# Ensure course-labs/r33-t001 is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t001"
work_dir = repo_root / "work" / "R33-T001"
sys.path.insert(0, str(lab_dir))

from pipeline import SchemaValidator, fixed_stratified_split, load_dataset


def run_checks() -> None:
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T001: DATA CONTRACT VALIDATOR")
    print("=" * 60)

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Capture Tool Versions
    tool_versions = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "git_version": subprocess.check_output(["git", "--version"], text=True).strip(),
    }
    with open(work_dir / "tool_versions.json", "w", encoding="utf-8") as f:
        json.dump(tool_versions, f, indent=2)
    print("[1/4] Tool versions recorded.")

    # 2. Run unittests via subprocess and save log
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(lab_dir), "-p", "test_*.py", "-v"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    unittest_log_path = work_dir / "unittest_output.txt"
    with open(unittest_log_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)

    if proc.returncode != 0:
        print(f"FAILED: unittest execution exited with code {proc.returncode}")
        print(proc.stdout)
        sys.exit(proc.returncode)
    print("[2/4] Unit tests passed successfully.")

    # 3. Evaluate Concrete Inputs vs Expected vs Actual Results
    fixture_path = lab_dir / "fixture.json"
    fixture_data = load_dataset(fixture_path)

    validator = SchemaValidator()
    valid_records, errors = validator.validate_records(fixture_data)

    train_set, test_set = fixed_stratified_split(fixture_data, train_ratio=0.5)

    test_cases_summary = []

    # Assertion 1: Fixture length
    assert len(fixture_data) == 8, f"Expected 8 records, got {len(fixture_data)}"
    test_cases_summary.append({
        "check": "fixture_record_count",
        "input": str(fixture_path),
        "expected": 8,
        "actual": len(fixture_data),
        "status": "PASS"
    })

    # Assertion 2: Feature x1 sequence
    expected_x1 = [0, 1, 2, 3, 4, 5, 6, 7]
    actual_x1 = [r["x1"] for r in fixture_data]
    assert actual_x1 == expected_x1, f"x1 mismatch: {actual_x1} != {expected_x1}"
    test_cases_summary.append({
        "check": "fixture_x1_sequence",
        "expected": expected_x1,
        "actual": actual_x1,
        "status": "PASS"
    })

    # Assertion 3: Feature x2 constant
    expected_x2 = [1] * 8
    actual_x2 = [r["x2"] for r in fixture_data]
    assert actual_x2 == expected_x2, f"x2 mismatch: {actual_x2} != {expected_x2}"
    test_cases_summary.append({
        "check": "fixture_x2_constant",
        "expected": expected_x2,
        "actual": actual_x2,
        "status": "PASS"
    })

    # Assertion 4: Labels
    expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    actual_labels = [r["label"] for r in fixture_data]
    assert actual_labels == expected_labels, f"Label mismatch: {actual_labels} != {expected_labels}"
    test_cases_summary.append({
        "check": "fixture_labels",
        "expected": expected_labels,
        "actual": actual_labels,
        "status": "PASS"
    })

    # Assertion 5: Valid records count
    assert len(valid_records) == 8 and len(errors) == 0, f"Validation failed on valid fixture: {errors}"
    test_cases_summary.append({
        "check": "clean_fixture_validation",
        "expected": {"valid_count": 8, "error_count": 0},
        "actual": {"valid_count": len(valid_records), "error_count": len(errors)},
        "status": "PASS"
    })

    # Assertion 6: Stratified split balance
    train_labels = [r["label"] for r in train_set]
    test_labels = [r["label"] for r in test_set]
    assert train_labels.count(0) == 2 and train_labels.count(1) == 2, "Train set imbalance"
    assert test_labels.count(0) == 2 and test_labels.count(1) == 2, "Test set imbalance"
    test_cases_summary.append({
        "check": "stratified_split_50_50",
        "expected": {"train_counts": {"0": 2, "1": 2}, "test_counts": {"0": 2, "1": 2}},
        "actual": {
            "train_counts": {"0": train_labels.count(0), "1": train_labels.count(1)},
            "test_counts": {"0": test_labels.count(0), "1": test_labels.count(1)}
        },
        "status": "PASS"
    })

    # 4. Save results to work/R33-T001/evidence.json
    evidence = {
        "task": "R33-T001",
        "student": "NAKSHATRA MESHRAM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "overall_status": "ALL_CHECKS_PASSED"
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Input vs Expected vs Actual evidence saved to work/R33-T001/evidence.json.")
    print("[4/4] ALL CHECKS PASSED SUCCESSFULLY (Exit code 0).")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
