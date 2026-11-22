"""
Verification runner for R33-T035: Make preprocessing explicit
=============================================================
Saves evidence.json, tool_versions.json, unittest_output.txt
in work/R33-T035/. Exits with code 1 on assertion failure.
"""

import json
import os
import subprocess
import sys
import platform
from datetime import datetime, timezone

WORK_DIR = os.path.join(os.path.dirname(__file__))
TASK_ID = "R33-T035"

def get_python_version():
    return platform.python_version()

def run_unittests():
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover",
         "-s", "course-labs/r33-t035", "-p", "test_*.py", "-v"],
        capture_output=True, text=True
    )
    return result.returncode, result.stdout + result.stderr

def save_file(filename, content):
    path = os.path.join(WORK_DIR, filename)
    with open(path, "w") as f:
        f.write(content if isinstance(content, str) else json.dumps(content, indent=2))
    print(f"  Saved: {path}")

def main():
    print(f"[{TASK_ID}] Running verification...")

    # --- Run unit tests ---
    exit_code, test_output = run_unittests()
    print(test_output)
    save_file("unittest_output.txt", test_output)

    # --- Tool versions ---
    tool_versions = {
        "task_id": TASK_ID,
        "python": get_python_version(),
        "platform": platform.platform(),
        "external_packages": "none — standard library only",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    save_file("tool_versions.json", tool_versions)

    # --- Evidence ---
    fixture = [
        {"record_id": "r1", "x1": 0, "x2": 0},
        {"record_id": "r2", "x1": 0, "x2": 1},
        {"record_id": "r3", "x1": 1, "x2": 0},
        {"record_id": "r4", "x1": 5, "x2": 5},
        {"record_id": "r5", "x1": 5, "x2": 6},
        {"record_id": "r6", "x1": 6, "x2": 5},
    ]

    sys.path.insert(0, "course-labs/r33-t035")
    from pipeline import (
        EXPECTED_ASSIGNMENTS, FIXTURE, FIXED_CENTROIDS,
        MinMaxScaler, nearest_centroid_assignments, run_pipeline
    )

    result = run_pipeline(FIXTURE, FIXTURE, FIXED_CENTROIDS)
    assignments = result["assignments"]

    evidence = {
        "task_id": TASK_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fixture": fixture,
        "fixed_centroids": FIXED_CENTROIDS,
        "inputs": {"rows": fixture, "centroids": list(FIXED_CENTROIDS)},
        "expected_assignments": EXPECTED_ASSIGNMENTS,
        "actual_assignments": assignments,
        "scaled_test": result["scaled_test"],
        "scaled_centroids": result["scaled_centroids"],
        "unit_test_exit_code": exit_code,
        "data_disclaimer": "Synthetic fixture only — not evidence of real-world performance."
    }
    save_file("evidence.json", evidence)

    # --- Final assertion ---
    assert assignments == EXPECTED_ASSIGNMENTS, (
        f"Assignment mismatch!\n  Expected: {EXPECTED_ASSIGNMENTS}\n  Got:      {assignments}"
    )
    assert exit_code == 0, f"Unit tests failed with exit code {exit_code}"

    print(f"\n[{TASK_ID}] ✅ All checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
