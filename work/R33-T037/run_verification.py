"""
Verification runner for R33-T037: Record reproducible configuration
===================================================================
Runs unit tests, evaluates configuration serialization round-trip,
and saves evidence.json, tool_versions.json, and unittest_output.txt.
Exits 0 on success, non-zero on failure.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_ID = "R33-T037"


def run_unittests():
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "course-labs/r33-t037",
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
    sys.path.insert(0, "course-labs/r33-t037")
    from pipeline import PipelineConfig, run_pipeline

    original_config = PipelineConfig(seed=42)
    serialized_json = original_config.to_json()
    reloaded_config = PipelineConfig.from_json(serialized_json)

    orig_result = run_pipeline(original_config)
    reloaded_result = run_pipeline(reloaded_config)

    evidence = {
        "task_id": TASK_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "serialized_configuration": original_config.to_dict(),
        "original_run": orig_result,
        "reloaded_run": reloaded_result,
        "identical_predictions": orig_result["test_predictions"] == reloaded_result["test_predictions"],
        "unit_test_exit_code": exit_code,
        "data_disclaimer": "Synthetic fixture only — not evidence of real-world performance.",
    }
    with open(os.path.join(WORK_DIR, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)

    # 4. Strict assertions
    assert exit_code == 0, f"Unit tests failed with code {exit_code}"
    assert orig_result["test_predictions"] == reloaded_result["test_predictions"], (
        "Reloaded configuration produced different test predictions!"
    )
    assert orig_result["val_predictions"] == reloaded_result["val_predictions"], (
        "Reloaded configuration produced different validation predictions!"
    )
    assert orig_result["train_ids"] == ["d1", "d2", "d3", "d4", "d5", "d6"], "Train split mismatch"
    assert orig_result["val_ids"] == ["d7", "d8"], "Val split mismatch"
    assert orig_result["test_ids"] == ["d9", "d10"], "Test split mismatch"

    print(f"[{TASK_ID}] ✅ All checks passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
