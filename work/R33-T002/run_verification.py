"""
Verification Runner for R33-T002: Deterministic Split Manifest.
Saves inputs, expected vs actual results, and tool versions into work/R33-T002/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t002"
work_dir = repo_root / "work" / "R33-T002"
sys.path.insert(0, str(lab_dir))

from pipeline import build_deterministic_split_manifest, load_dataset


def run_checks() -> None:
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T002: DETERMINISTIC SPLIT MANIFEST")
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
    records = load_dataset(fixture_path)

    # Assertion 1: Exact record count (6)
    assert len(records) == 6, f"Expected 6 records, got {len(records)}"

    # Assertion 2: Formula target = 2*x1 + 1
    expected_x1 = [0, 1, 2, 3, 4, 5]
    expected_x2 = [1, 1, 1, 1, 1, 1]
    expected_targets = [1, 3, 5, 7, 9, 11]
    expected_ids = ["r1", "r2", "r3", "r4", "r5", "r6"]

    actual_x1 = [r["x1"] for r in records]
    actual_x2 = [r["x2"] for r in records]
    actual_targets = [r["target"] for r in records]
    actual_ids = [r["record_id"] for r in records]

    assert actual_x1 == expected_x1, f"x1 mismatch: {actual_x1} != {expected_x1}"
    assert actual_x2 == expected_x2, f"x2 mismatch: {actual_x2} != {expected_x2}"
    assert actual_targets == expected_targets, f"Target mismatch: {actual_targets} != {expected_targets}"
    assert actual_ids == expected_ids, f"ID mismatch: {actual_ids} != {expected_ids}"

    # Assertion 3: Manifest generation and exact partition IDs
    manifest = build_deterministic_split_manifest(records)
    assert manifest.train_record_ids == ["r1", "r2", "r3", "r4"], "Train IDs mismatch"
    assert manifest.validation_record_ids == ["r5"], "Val ID mismatch"
    assert manifest.test_record_ids == ["r6"], "Test ID mismatch"
    assert manifest.is_disjoint is True, "Disjointness violated"

    # Assertion 4: Manifest repeatability
    manifest_repeat = build_deterministic_split_manifest(records)
    assert manifest.to_dict() == manifest_repeat.to_dict(), "Manifest is not deterministic"

    test_cases_summary = [
        {
            "check": "fixture_record_count",
            "expected": 6,
            "actual": len(records),
            "status": "PASS"
        },
        {
            "check": "fixture_linear_formula_evaluation",
            "formula": "target = 2 * x1 + 1",
            "expected_targets": expected_targets,
            "actual_targets": actual_targets,
            "status": "PASS"
        },
        {
            "check": "manifest_partition_assignment",
            "expected": {
                "train": ["r1", "r2", "r3", "r4"],
                "validation": ["r5"],
                "test": ["r6"]
            },
            "actual": {
                "train": manifest.train_record_ids,
                "validation": manifest.validation_record_ids,
                "test": manifest.test_record_ids
            },
            "status": "PASS"
        },
        {
            "check": "partition_disjointness_verification",
            "expected_is_disjoint": True,
            "actual_is_disjoint": manifest.is_disjoint,
            "status": "PASS"
        },
        {
            "check": "manifest_determinism_repeat_run",
            "status": "PASS"
        }
    ]

    evidence = {
        "task": "R33-T002",
        "student": "NAKSHATRA MESHRAM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "manifest": manifest.to_dict(),
        "overall_status": "ALL_CHECKS_PASSED"
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Input vs Expected vs Actual evidence saved to work/R33-T002/evidence.json.")
    print("[4/4] ALL CHECKS PASSED SUCCESSFULLY (Exit code 0).")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
