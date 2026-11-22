"""
Verification Runner for Course Lab R33-T034.
Executes automated checks, records tool versions, and saves inputs,
expected vs actual results, and unit test logs into work/R33-T034/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

# Ensure course-labs/r33-t034 is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t034"
work_dir = repo_root / "work" / "R33-T034"
sys.path.insert(0, str(lab_dir))

from pipeline import (
    SchemaValidator,
    calculate_accuracy,
    calculate_f1_score,
    calculate_mae,
    calculate_precision,
    calculate_recall,
    calculate_token_exact_match,
    load_dataset,
)


def run_checks() -> None:
    print("=" * 65)
    print("RUNNING VERIFICATION CHECKS: R33-T034 EVALUATION METRIC")
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
    records = load_dataset(fixture_path)
    validator = SchemaValidator()
    valid_recs, val_errors = validator.validate_records(records)

    test_cases_summary = []

    # Check 1: Classification precision/recall on truth[1,1,0,0]/pred[1,0,1,0]
    y_true_clf = [1, 1, 0, 0]
    y_pred_clf = [1, 0, 1, 0]
    prec = calculate_precision(y_true_clf, y_pred_clf)
    rec = calculate_recall(y_true_clf, y_pred_clf)
    assert abs(prec - 0.5) < 1e-6 and abs(rec - 0.5) < 1e-6, "Classification precision/recall assertion failed"
    test_cases_summary.append({
        "check": "classification_precision_recall_case",
        "inputs": {"y_true": y_true_clf, "y_pred": y_pred_clf},
        "expected": {"precision": 0.5, "recall": 0.5},
        "actual": {"precision": prec, "recall": rec},
        "status": "PASS",
    })

    # Check 2: Regression MAE on truth[1,2,3]/pred[1,3,2]
    y_true_reg = [1.0, 2.0, 3.0]
    y_pred_reg = [1.0, 3.0, 2.0]
    mae = calculate_mae(y_true_reg, y_pred_reg)
    expected_mae = 2.0 / 3.0
    assert abs(mae - expected_mae) < 1e-6, "Regression MAE assertion failed"
    test_cases_summary.append({
        "check": "regression_mae_case",
        "inputs": {"y_true": y_true_reg, "y_pred": y_pred_reg},
        "expected": {"mae": expected_mae},
        "actual": {"mae": mae},
        "status": "PASS",
    })

    # Check 3: Zero-denominator safety
    zero_div_prec = calculate_precision([1, 1], [0, 0], zero_division=0.0)
    zero_div_rec = calculate_recall([0, 0], [1, 1], zero_division=0.0)
    assert zero_div_prec == 0.0 and zero_div_rec == 0.0, "Zero denominator check failed"
    test_cases_summary.append({
        "check": "zero_denominator_handling",
        "expected": {"precision_zero_denom": 0.0, "recall_zero_denom": 0.0},
        "actual": {"precision_zero_denom": zero_div_prec, "recall_zero_denom": zero_div_rec},
        "status": "PASS",
    })

    # Check 4: Text fixture verification (t1..t6)
    expected_ids = ["t1", "t2", "t3", "t4", "t5", "t6"]
    expected_texts = ["red apple", "green apple", "blue sky", "dark sky", "red fruit", "blue cloud"]
    expected_labels = [1, 1, 0, 0, 1, 0]
    actual_ids = [r["record_id"] for r in records]
    actual_texts = [r["text"] for r in records]
    actual_labels = [r["label"] for r in records]
    assert actual_ids == expected_ids and actual_texts == expected_texts and actual_labels == expected_labels, "Fixture mismatch"
    test_cases_summary.append({
        "check": "synthetic_text_fixture_exactness",
        "expected": {"record_count": 6, "ids": expected_ids, "labels": expected_labels},
        "actual": {"record_count": len(records), "ids": actual_ids, "labels": actual_labels},
        "status": "PASS",
    })

    # Check 5: Schema validation pass
    assert len(valid_recs) == 6 and len(val_errors) == 0, "Schema validation error"
    test_cases_summary.append({
        "check": "fixture_schema_validation",
        "expected": {"valid_count": 6, "error_count": 0},
        "actual": {"valid_count": len(valid_recs), "error_count": len(val_errors)},
        "status": "PASS",
    })

    # 4. Save results to evidence.json
    evidence = {
        "task": "R33-T034",
        "student": "Snehakumari Vinod Singh",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "overall_status": "ALL_CHECKS_PASSED",
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Input vs Expected vs Actual evidence saved to work/R33-T034/evidence.json.")
    print("[4/4] ALL CHECKS PASSED SUCCESSFULLY (Exit code 0).")
    print("=" * 65)


if __name__ == "__main__":
    run_checks()
