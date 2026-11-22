"""
Verification Runner for Course Lab R33-T031.
Executes automated checks, records tool versions, and saves inputs,
expected vs actual results, and unit test logs into work/R33-T031/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

# Ensure course-labs/r33-t031 is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t031"
work_dir = repo_root / "work" / "R33-T031"
sys.path.insert(0, str(lab_dir))

from pipeline import SchemaValidator, fixed_stratified_split, load_dataset


def run_checks() -> None:
    print("=" * 65)
    print("RUNNING VERIFICATION CHECKS: R33-T031 IMAGE DATA CONTRACT")
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
    print(f"[2/4] Unit tests passed successfully. Log written to unittest_output.txt.")

    # 3. Evaluate Concrete Inputs vs Expected vs Actual Results
    fixture_path = lab_dir / "fixture.json"
    fixture_data = load_dataset(fixture_path)

    validator = SchemaValidator()
    valid_records, errors = validator.validate_records(fixture_data)
    train_set, test_set = fixed_stratified_split(fixture_data, train_ratio=0.5)

    test_cases_summary = []

    # Check 1: Record count
    assert len(fixture_data) == 6, f"Expected 6 records, got {len(fixture_data)}"
    test_cases_summary.append({
        "check": "fixture_record_count",
        "input": str(fixture_path),
        "expected": 6,
        "actual": len(fixture_data),
        "status": "PASS"
    })

    # Check 2: All-zero images count (label 0)
    zero_records = [r for r in fixture_data if r["label"] == 0]
    assert len(zero_records) == 3, f"Expected 3 label-0 images, got {len(zero_records)}"
    for r in zero_records:
        pixels = r["pixels"]
        assert all(all(p == 0 for p in row) for row in pixels), f"Non-zero pixel in record {r['record_id']}"
    test_cases_summary.append({
        "check": "all_zero_images_label_0",
        "expected": {"count": 3, "label": 0, "matrix": [[0]*4]*4},
        "actual": {"count": len(zero_records), "label": 0, "all_pixels_zero": True},
        "status": "PASS"
    })

    # Check 3: White diagonal images count (label 1)
    diag_records = [r for r in fixture_data if r["label"] == 1]
    assert len(diag_records) == 3, f"Expected 3 label-1 images, got {len(diag_records)}"
    for r in diag_records:
        pixels = r["pixels"]
        for i in range(4):
            for j in range(4):
                expected_p = 255 if i == j else 0
                assert pixels[i][j] == expected_p, f"Pixel mismatch in diagonal image at [{i},{j}]"
    test_cases_summary.append({
        "check": "white_diagonal_images_label_1",
        "expected": {"count": 3, "label": 1, "diagonal_val": 255, "off_diagonal_val": 0},
        "actual": {"count": len(diag_records), "label": 1, "diagonal_correct": True},
        "status": "PASS"
    })

    # Check 4: Clean fixture validation pass
    assert len(valid_records) == 6 and len(errors) == 0, f"Clean fixture produced errors: {errors}"
    test_cases_summary.append({
        "check": "clean_fixture_validation",
        "expected": {"valid_count": 6, "error_count": 0},
        "actual": {"valid_count": len(valid_records), "error_count": len(errors)},
        "status": "PASS"
    })

    # Check 5: Duplicate ID detection
    dup_test_records = [
        {"record_id": "dup_01", "group_id": "g1", "pixels": [[0]*4]*4, "label": 0},
        {"record_id": "dup_01", "group_id": "g1", "pixels": [[0]*4]*4, "label": 0},
    ]
    _, dup_errors = validator.validate_records(dup_test_records)
    assert any(e.field == "record_id" and "Duplicate" in e.reason for e in dup_errors), "Failed to catch duplicate ID"
    test_cases_summary.append({
        "check": "duplicate_id_detection",
        "expected": "ValidationError on record_id with duplicate reason",
        "actual": "Detected duplicate record_id",
        "status": "PASS"
    })

    # Check 6: Stratified train/test split disjointness
    train_ids = {r["record_id"] for r in train_set}
    test_ids = {r["record_id"] for r in test_set}
    assert len(train_ids.intersection(test_ids)) == 0, "Train and test splits overlap"
    test_cases_summary.append({
        "check": "stratified_split_disjointness",
        "expected": {"overlap_count": 0},
        "actual": {"overlap_count": len(train_ids.intersection(test_ids))},
        "status": "PASS"
    })

    # 4. Save results to evidence.json
    evidence = {
        "task": "R33-T031",
        "student": "Snehakumari Vinod Singh",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "overall_status": "ALL_CHECKS_PASSED"
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Input vs Expected vs Actual evidence saved to work/R33-T031/evidence.json.")
    print("[4/4] ALL CHECKS PASSED SUCCESSFULLY (Exit code 0).")
    print("=" * 65)


if __name__ == "__main__":
    run_checks()
