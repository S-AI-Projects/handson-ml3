"""
Verification runner for R33-T040: Define a saved-pipeline interface
===================================================================
Runs unit tests, verifies save/reload roundtrip and tolerance agreement,
and records evidence.json, tool_versions.json, and unittest_output.txt.
Exits 0 on success, non-zero on failure.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_ID = "R33-T040"


def run_unittests():
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "course-labs/r33-t040",
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
    sys.path.insert(0, "course-labs/r33-t040")
    from pipeline import run_saved_pipeline_roundtrip

    roundtrip_result = run_saved_pipeline_roundtrip(tolerance=1e-6)

    evidence = {
        "task_id": TASK_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "roundtrip_result": roundtrip_result,
        "unit_test_exit_code": exit_code,
        "data_disclaimer": "Synthetic fixture only — not evidence of real-world performance.",
    }
    with open(os.path.join(WORK_DIR, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)

    # 4. Strict assertions
    assert exit_code == 0, f"Unit tests failed with code {exit_code}"
    assert roundtrip_result["status"] == "PASS", "Roundtrip verification failed"
    assert roundtrip_result["requires_training_data_at_inference"] is False, "Training data leaked to inference"
    assert roundtrip_result["requires_credentials"] is False, "Credentials required at inference"

    for rid in roundtrip_result["predictions_before"]:
        p_b = roundtrip_result["predictions_before"][rid]["probability"]
        p_a = roundtrip_result["predictions_after"][rid]["probability"]
        assert abs(p_b - p_a) <= 1e-6, f"Prediction discrepancy on {rid}"

    print(f"[{TASK_ID}] ✅ All checks passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
