"""
Verification Runner for R33-T005: Preprocessing Contract & Parameter Isolation.
Saves inputs, expected vs actual results, and tool versions into work/R33-T005/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t005"
work_dir = repo_root / "work" / "R33-T005"
sys.path.insert(0, str(lab_dir))

from pipeline import ExplicitPreprocessor, load_dataset


def run_checks() -> None:
    print("=" * 60)
    print("RUNNING CHECKS FOR R33-T005: PREPROCESSING CONTRACT")
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
    train_records = [r for r in records if r.get("split") == "train"]
    val_records = [r for r in records if r.get("split") == "validation"]
    test_records = [r for r in records if r.get("split") == "test"]

    assert len(records) == 6, f"Expected 6 records, got {len(records)}"
    assert len(train_records) == 4, f"Expected 4 train records, got {len(train_records)}"

    preprocessor = ExplicitPreprocessor(feature_order=("x1", "x2"))
    preprocessor.fit(train_records)

    expected_mean_x1 = 1.5
    expected_std_x1 = math.sqrt(1.25)

    assert abs(preprocessor.state_.means["x1"] - expected_mean_x1) < 1e-6, "Mean x1 mismatch"
    assert abs(preprocessor.state_.stds["x1"] - expected_std_x1) < 1e-6, "Std x1 mismatch"

    # Transform
    train_out = preprocessor.transform(train_records)
    val_out = preprocessor.transform(val_records)
    test_out = preprocessor.transform(test_records)

    # Assert shape and dtypes
    for out_batch in (train_out, val_out, test_out):
        for row in out_batch:
            vec = row["_feature_vector"]
            assert len(vec) == 2, "Feature vector shape must be 2"
            assert isinstance(vec[0], float) and isinstance(vec[1], float), "Dtype must be float"

    # Assert expected transformed values
    expected_r1_x1 = (0 - 1.5) / expected_std_x1
    expected_r5_x1 = (4 - 1.5) / expected_std_x1
    expected_r6_x1 = (5 - 1.5) / expected_std_x1

    assert abs(train_out[0]["x1"] - expected_r1_x1) < 1e-6
    assert abs(val_out[0]["x1"] - expected_r5_x1) < 1e-6
    assert abs(test_out[0]["x1"] - expected_r6_x1) < 1e-6

    test_cases_summary = [
        {
            "check": "fixture_record_count_and_partitions",
            "expected": {"total": 6, "train": 4, "validation": 1, "test": 1},
            "actual": {"total": len(records), "train": len(train_records), "validation": len(val_records), "test": len(test_records)},
            "status": "PASS"
        },
        {
            "check": "training_fitted_statistics",
            "expected": {"mean_x1": expected_mean_x1, "std_x1": expected_std_x1},
            "actual": {"mean_x1": preprocessor.state_.means["x1"], "std_x1": preprocessor.state_.stds["x1"]},
            "status": "PASS"
        },
        {
            "check": "transformed_output_shape_and_dtype",
            "expected_shape": 2,
            "expected_dtype": "float",
            "status": "PASS"
        },
        {
            "check": "held_out_validation_standardized_x1",
            "expected_val_x1": expected_r5_x1,
            "actual_val_x1": val_out[0]["x1"],
            "status": "PASS"
        },
        {
            "check": "held_out_test_standardized_x1",
            "expected_test_x1": expected_r6_x1,
            "actual_test_x1": test_out[0]["x1"],
            "status": "PASS"
        }
    ]

    evidence = {
        "task": "R33-T005",
        "student": "NAKSHATRA MESHRAM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "fitted_state": preprocessor.state_.to_dict(),
        "overall_status": "ALL_CHECKS_PASSED"
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Evidence saved to work/R33-T005/evidence.json.")
    print("[4/4] ALL CHECKS PASSED (Exit code 0).")
    print("=" * 60)


if __name__ == "__main__":
    run_checks()
