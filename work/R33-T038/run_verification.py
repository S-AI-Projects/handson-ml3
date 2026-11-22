"""
Verification runner for R33-T038: Explain an error case
========================================================
Runs unit tests, evaluates error analysis pipeline, and saves
evidence.json, tool_versions.json, and unittest_output.txt.
Exits 0 on success, non-zero on failure.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_ID = "R33-T038"


def run_unittests():
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "course-labs/r33-t038",
        "-p",
        "test_*.py",
        "-v",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr


def main():
    print(f"[{TASK_ID}] Running verification...")

    # 1. Run unit tests
    exit_code, test_output = run_unittests()
    print(test_output)

    with open(os.path.join(WORK_DIR, "unittest_output.txt"), "w") as f:
        f.write(test_output)

    # 2. Tool versions
    tool_versions = {
        "task_id": TASK_ID,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "external_packages": "none — standard library only",
        "device": "CPU-only",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(WORK_DIR, "tool_versions.json"), "w") as f:
        json.dump(tool_versions, f, indent=2)

    # 3. Evidence
    sys.path.insert(0, "course-labs/r33-t038")
    from pipeline import run_error_analysis_pipeline

    pipeline_output = run_error_analysis_pipeline()
    error_report = pipeline_output["error_report"]

    evidence = {
        "task_id": TASK_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "injected_error_details": {
            "target_record_id": pipeline_output["injected_target"],
            "reason_category": pipeline_output["injected_reason"],
        },
        "clean_report_summary": {
            "total_records": pipeline_output["clean_report"]["total_records"],
            "total_errors": pipeline_output["clean_report"]["total_errors"],
            "accuracy": pipeline_output["clean_report"]["accuracy"],
        },
        "error_report": error_report,
        "unit_test_exit_code": exit_code,
        "data_disclaimer": "Synthetic fixture only — not evidence of real-world performance.",
    }
    with open(os.path.join(WORK_DIR, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)

    # 4. Strict assertions
    assert exit_code == 0, f"Unit tests failed with code {exit_code}"
    assert error_report["total_errors"] == 1, "Expected exactly 1 error in report"
    assert len(error_report["error_rows"]) == 1, "Detailed rows count mismatch"
    assert error_report["error_rows"][0]["record_id"] == "s4", "Wrong record ID in error report"
    assert error_report["error_rows"][0]["expected_value"] == 0, "Wrong expected value"
    assert error_report["error_rows"][0]["predicted_value"] == 1, "Wrong predicted value"
    assert error_report["error_rows"][0]["reason_category"] == "decision_boundary_proximity", "Wrong reason"

    print(f"[{TASK_ID}] ✅ All checks passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
