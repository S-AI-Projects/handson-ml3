"""
Verification Runner for R33-T004: Metric Correctness & Zero-Denominator Handling.
Saves inputs, expected vs actual results, and tool versions into work/R33-T004/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t004"
work_dir = repo_root / "work" / "R33-T004"
sys.path.insert(0, str(lab_dir))

from pipeline import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    DecisionStumpClassifier,
    load_dataset,
)


def run_checks() -> None:
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T004: METRIC CORRECTNESS")
    print("=" * 60)

    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tool Versions
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
    # Canonical Case 1: Classification truth=[1,1,0,0], pred=[1,0,1,0]
    clf_m = calculate_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert abs(clf_m.precision - 0.5) < 1e-6, f"Expected precision 0.5, got {clf_m.precision}"
    assert abs(clf_m.recall - 0.5) < 1e-6, f"Expected recall 0.5, got {clf_m.recall}"
    assert abs(clf_m.accuracy - 0.5) < 1e-6, f"Expected accuracy 0.5, got {clf_m.accuracy}"

    # Canonical Case 2: Regression truth=[1,2,3], pred=[1,3,2]
    reg_m = calculate_regression_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])
    expected_mae = 2.0 / 3.0
    assert abs(reg_m.mae - expected_mae) < 1e-6, f"Expected MAE 2/3, got {reg_m.mae}"

    # Zero-denominator check
    zero_div_m = calculate_classification_metrics([1, 1], [0, 0], zero_division=0.0)
    assert zero_div_m.precision == 0.0, "Expected precision 0.0 on zero division"
    assert zero_div_m.zero_division_applied is True, "Expected zero_division_applied=True"

    # Fixture verification
    fixture_path = lab_dir / "fixture.json"
    records = load_dataset(fixture_path)
    assert len(records) == 8, f"Expected 8 records, got {len(records)}"

    stump = DecisionStumpClassifier(threshold=3.5)
    stump_m = stump.evaluate(records)
    assert stump_m.accuracy == 1.0, f"Expected accuracy 1.0, got {stump_m.accuracy}"

    test_cases_summary = [
        {
            "check": "canonical_classification_metrics",
            "truth": [1, 1, 0, 0],
            "prediction": [1, 0, 1, 0],
            "expected": {"precision": 0.5, "recall": 0.5, "accuracy": 0.5, "f1_score": 0.5},
            "actual": clf_m.to_dict(),
            "status": "PASS"
        },
        {
            "check": "canonical_regression_metrics",
            "truth": [1.0, 2.0, 3.0],
            "prediction": [1.0, 3.0, 2.0],
            "expected_mae": expected_mae,
            "actual_mae": reg_m.mae,
            "status": "PASS"
        },
        {
            "check": "zero_denominator_safeguard",
            "expected_precision": 0.0,
            "actual_precision": zero_div_m.precision,
            "zero_division_applied": zero_div_m.zero_division_applied,
            "status": "PASS"
        },
        {
            "check": "synthetic_fixture_decision_stump",
            "fixture_count": len(records),
            "stump_threshold": 3.5,
            "expected_accuracy": 1.0,
            "actual_accuracy": stump_m.accuracy,
            "status": "PASS"
        }
    ]

    evidence = {
        "task": "R33-T004",
        "student": "NAKSHATRA MESHRAM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "overall_status": "ALL_CHECKS_PASSED"
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Evidence saved to work/R33-T004/evidence.json.")
    print("[4/4] ALL CHECKS PASSED (Exit code 0).")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
