"""
Verification Runner for Course Lab R33-T032.
Executes automated checks, records tool versions, and saves inputs,
expected vs actual results, and unit test logs into work/R33-T032/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

# Ensure course-labs/r33-t032 is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t032"
work_dir = repo_root / "work" / "R33-T032"
sys.path.insert(0, str(lab_dir))

from pipeline import (
    SchemaValidator,
    SplitManifest,
    generate_chronological_split_manifest,
    generate_stratified_split_manifest,
    load_dataset,
    validate_split_manifest,
)


def run_checks() -> None:
    print("=" * 65)
    print("RUNNING VERIFICATION CHECKS: R33-T032 LEAKAGE-FREE SPLITTING")
    print("=" * 65)

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
    print("[1/4] Tool versions recorded in tool_versions.json.")

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
    print("[2/4] Unit tests passed successfully. Log written to unittest_output.txt.")

    # 3. Evaluate Concrete Inputs vs Expected vs Actual Results
    fixture_path = lab_dir / "fixture.json"
    manifest_path = lab_dir / "split_manifest.json"
    records = load_dataset(fixture_path)
    manifest = SplitManifest.load(manifest_path)

    validator = SchemaValidator()
    valid_records, errors = validator.validate_records(records)

    test_cases_summary = []

    # Check 1: Record count
    assert len(records) == 8, f"Expected 8 records, got {len(records)}"
    test_cases_summary.append({
        "check": "fixture_record_count",
        "input": str(fixture_path),
        "expected": 8,
        "actual": len(records),
        "status": "PASS",
    })

    # Check 2: Feature x1 sequence [0..7]
    expected_x1 = [0, 1, 2, 3, 4, 5, 6, 7]
    actual_x1 = [r["x1"] for r in records]
    assert actual_x1 == expected_x1, f"x1 mismatch: {actual_x1} != {expected_x1}"
    test_cases_summary.append({
        "check": "fixture_x1_sequence",
        "expected": expected_x1,
        "actual": actual_x1,
        "status": "PASS",
    })

    # Check 3: Feature x2 constant [1..1]
    expected_x2 = [1] * 8
    actual_x2 = [r["x2"] for r in records]
    assert actual_x2 == expected_x2, f"x2 mismatch: {actual_x2} != {expected_x2}"
    test_cases_summary.append({
        "check": "fixture_x2_constant",
        "expected": expected_x2,
        "actual": actual_x2,
        "status": "PASS",
    })

    # Check 4: Labels [0,0,0,0,1,1,1,1]
    expected_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    actual_labels = [r["label"] for r in records]
    assert actual_labels == expected_labels, f"Label mismatch: {actual_labels} != {expected_labels}"
    test_cases_summary.append({
        "check": "fixture_labels",
        "expected": expected_labels,
        "actual": actual_labels,
        "status": "PASS",
    })

    # Check 5: Disjoint split sets
    s_train = set(manifest.train)
    s_val = set(manifest.validation)
    s_test = set(manifest.test)
    assert len(s_train.intersection(s_val)) == 0, "Train and Val overlap"
    assert len(s_train.intersection(s_test)) == 0, "Train and Test overlap"
    assert len(s_val.intersection(s_test)) == 0, "Val and Test overlap"
    test_cases_summary.append({
        "check": "disjoint_split_partitions",
        "expected": {"overlaps": 0},
        "actual": {"train_val_overlap": 0, "train_test_overlap": 0, "val_test_overlap": 0},
        "status": "PASS",
    })

    # Check 6: Deterministic repeat generation
    run2 = generate_stratified_split_manifest(records, train_ratio=0.5, seed=42)
    assert manifest.train == run2.train and manifest.validation == run2.validation and manifest.test == run2.test, "Manifest generation jitter"
    test_cases_summary.append({
        "check": "deterministic_manifest_regeneration",
        "expected": "Exact match across repeated runs",
        "actual": "Identical manifest produced",
        "status": "PASS",
    })

    # Check 7: Chronological time-series ordering
    chrono = generate_chronological_split_manifest(records, train_ratio=0.5, val_ratio=0.25)
    chrono_violations = validate_split_manifest(chrono, records, check_chronological=True)
    assert len(chrono_violations) == 0, f"Chronological split failed: {chrono_violations}"
    test_cases_summary.append({
        "check": "chronological_split_monotonicity",
        "expected": "Zero temporal lookahead violations",
        "actual": "Train <= Validation <= Test monotonic ordering verified",
        "status": "PASS",
    })

    # 4. Save results to evidence.json
    evidence = {
        "task": "R33-T032",
        "student": "Snehakumari Vinod Singh",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "manifest_summary": manifest.to_dict()["summary"],
        "overall_status": "ALL_CHECKS_PASSED",
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Input vs Expected vs Actual evidence saved to work/R33-T032/evidence.json.")
    print("[4/4] ALL CHECKS PASSED SUCCESSFULLY (Exit code 0).")
    print("=" * 65)


if __name__ == "__main__":
    run_checks()
