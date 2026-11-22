"""
Verification runner for R33-T039: Test an input perturbation
=============================================================
Runs unit tests, verifies perturbation equivalence and fallback handling,
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
TASK_ID = "R33-T039"


def run_unittests():
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "course-labs/r33-t039",
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
    sys.path.insert(0, "course-labs/r33-t039")
    from pipeline import run_perturbation_pipeline

    perturbation_result = run_perturbation_pipeline()

    evidence = {
        "task_id": TASK_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "perturbation_result": perturbation_result,
        "unit_test_exit_code": exit_code,
        "data_disclaimer": "Synthetic fixture only — not evidence of real-world performance.",
    }
    with open(os.path.join(WORK_DIR, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)

    # 4. Strict assertions
    assert exit_code == 0, f"Unit tests failed with code {exit_code}"
    assert perturbation_result["status"] == "PASS", "Perturbation pipeline failed"
    assert perturbation_result["id_aligned_equivalent"] is True, "Permutation equivalence failed"
    assert perturbation_result["unseen_token_fallback"] == "mapped_to_unk", "Unseen token fallback failed"
    assert perturbation_result["missing_field_rejection"] == "rejected_with_ValueError", "Rejection check failed"

    print(f"[{TASK_ID}] ✅ All checks passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
