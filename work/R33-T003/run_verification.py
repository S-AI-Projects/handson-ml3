"""
Verification Runner for R33-T003: Minimal Majority-Class Baseline.
Saves inputs, expected vs actual results, and tool versions into work/R33-T003/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t003"
work_dir = repo_root / "work" / "R33-T003"
sys.path.insert(0, str(lab_dir))

from pipeline import MajorityClassBaseline, load_dataset, split_train_test


def run_checks() -> None:
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T003: MINIMAL MAJORITY BASELINE")
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
    fixture_path = lab_dir / "fixture.json"
    records = load_dataset(fixture_path)

    # Check 1: Record Count
    assert len(records) == 6, f"Expected 6 records, got {len(records)}"

    # Check 2: Three all-zero (0) and three diagonal (1)
    zero_rows = [r for r in records if r["label"] == 0]
    diag_rows = [r for r in records if r["label"] == 1]
    assert len(zero_rows) == 3, f"Expected 3 zero rows, got {len(zero_rows)}"
    assert len(diag_rows) == 3, f"Expected 3 diagonal rows, got {len(diag_rows)}"

    for z in zero_rows:
        assert z["pixels"] == [0] * 16, "Zero image pixels not all 0"

    expected_diag_px = [255 if idx in (0, 5, 10, 15) else 0 for idx in range(16)]
    for d in diag_rows:
        assert d["pixels"] == expected_diag_px, "Diagonal image pixels mismatch"

    # Check 3: Baseline model fit & held-out evaluation
    train_set, test_set = split_train_test(
        records,
        train_ids=["img_01", "img_02", "img_03", "img_04"],
        test_ids=["img_05", "img_06"],
    )

    baseline = MajorityClassBaseline()
    baseline.fit(train_set)
    assert baseline.majority_class == 0, f"Expected majority class 0, got {baseline.majority_class}"

    eval_result = baseline.evaluate(test_set)
    assert eval_result.test_sample_count == 2
    assert eval_result.correct_predictions == 0
    assert eval_result.accuracy == 0.0

    test_cases_summary = [
        {
            "check": "fixture_record_count",
            "expected": 6,
            "actual": len(records),
            "status": "PASS"
        },
        {
            "check": "class_distribution_3_zeroes_3_diagonals",
            "expected": {"label_0": 3, "label_1": 3},
            "actual": {"label_0": len(zero_rows), "label_1": len(diag_rows)},
            "status": "PASS"
        },
        {
            "check": "baseline_parameter_training_isolation",
            "expected_majority_class": 0,
            "actual_majority_class": baseline.majority_class,
            "status": "PASS"
        },
        {
            "check": "held_out_test_evaluation",
            "expected_test_accuracy": 0.0,
            "actual_test_accuracy": eval_result.accuracy,
            "status": "PASS"
        }
    ]

    evidence = {
        "task": "R33-T003",
        "student": "NAKSHATRA MESHRAM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "evaluation_summary": eval_result.to_dict(),
        "overall_status": "ALL_CHECKS_PASSED"
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Evidence saved to work/R33-T003/evidence.json.")
    print("[4/4] ALL CHECKS PASSED (Exit code 0).")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
