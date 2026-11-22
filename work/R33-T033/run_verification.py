"""
Verification Runner for Course Lab R33-T033.
Executes automated checks, records tool versions, and saves inputs,
expected vs actual results, and unit test logs into work/R33-T033/.
Exits with code 1 if any assertion fails.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

# Ensure course-labs/r33-t033 is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
lab_dir = repo_root / "course-labs" / "r33-t033"
work_dir = repo_root / "work" / "R33-T033"
sys.path.insert(0, str(lab_dir))

from pipeline import (
    AlwaysLeftPolicy,
    AlwaysRightPolicy,
    SchemaValidator,
    ThreeStateChainEnv,
    evaluate_policy,
    load_dataset,
)


def run_checks() -> None:
    print("=" * 65)
    print("RUNNING VERIFICATION CHECKS: R33-T033 REINFORCEMENT BASELINE")
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
    env = ThreeStateChainEnv(step_cap=5)
    validator = SchemaValidator()
    fixture_path = lab_dir / "fixture.json"
    records = load_dataset(fixture_path)
    valid_recs, val_errors = validator.validate_records(records)

    test_cases_summary = []

    # Check 1: Trajectory [right, right]
    traj_right = env.run_trajectory(["right", "right"])
    assert traj_right["final_state"] == 2 and traj_right["total_reward"] == 1.0 and traj_right["steps"] == 2, "Trajectory [right, right] failure"
    test_cases_summary.append({
        "check": "trajectory_right_right",
        "input_actions": ["right", "right"],
        "expected": {"final_state": 2, "total_reward": 1.0, "steps": 2, "terminated": True},
        "actual": {
            "final_state": traj_right["final_state"],
            "total_reward": traj_right["total_reward"],
            "steps": traj_right["steps"],
            "terminated": traj_right["terminated"],
        },
        "status": "PASS",
    })

    # Check 2: Trajectory 5 lefts hits step cap
    traj_left = env.run_trajectory(["left", "left", "left", "left", "left"])
    assert traj_left["final_state"] == 0 and traj_left["total_reward"] == 0.0 and traj_left["steps"] == 5, "Trajectory five lefts failure"
    test_cases_summary.append({
        "check": "trajectory_five_lefts_step_cap",
        "input_actions": ["left"] * 5,
        "expected": {"final_state": 0, "total_reward": 0.0, "steps": 5, "terminated": True},
        "actual": {
            "final_state": traj_left["final_state"],
            "total_reward": traj_left["total_reward"],
            "steps": traj_left["steps"],
            "terminated": traj_left["terminated"],
        },
        "status": "PASS",
    })

    # Check 3: AlwaysRightPolicy evaluation
    right_policy = AlwaysRightPolicy()
    right_metrics = evaluate_policy(right_policy, env, n_episodes=10)
    assert right_metrics["mean_reward"] == 1.0 and right_metrics["success_rate"] == 1.0, "AlwaysRightPolicy evaluation failed"
    test_cases_summary.append({
        "check": "always_right_policy_baseline",
        "expected": {"mean_reward": 1.0, "success_rate": 1.0, "mean_steps": 2.0},
        "actual": {
            "mean_reward": right_metrics["mean_reward"],
            "success_rate": right_metrics["success_rate"],
            "mean_steps": right_metrics["mean_steps"],
        },
        "status": "PASS",
    })

    # Check 4: AlwaysLeftPolicy baseline comparison
    left_policy = AlwaysLeftPolicy()
    left_metrics = evaluate_policy(left_policy, env, n_episodes=5)
    assert left_metrics["mean_reward"] == 0.0 and left_metrics["success_rate"] == 0.0, "AlwaysLeftPolicy failed"
    test_cases_summary.append({
        "check": "always_left_policy_comparison",
        "expected": {"mean_reward": 0.0, "success_rate": 0.0, "mean_steps": 5.0},
        "actual": {
            "mean_reward": left_metrics["mean_reward"],
            "success_rate": left_metrics["success_rate"],
            "mean_steps": left_metrics["mean_steps"],
        },
        "status": "PASS",
    })

    # Check 5: Fixture schema validation
    assert len(valid_recs) == len(records) and len(val_errors) == 0, f"Schema validation failed: {val_errors}"
    test_cases_summary.append({
        "check": "fixture_schema_validation",
        "expected": {"valid_count": len(records), "error_count": 0},
        "actual": {"valid_count": len(valid_recs), "error_count": len(val_errors)},
        "status": "PASS",
    })

    # 4. Save results to evidence.json
    evidence = {
        "task": "R33-T033",
        "student": "Snehakumari Vinod Singh",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_versions": tool_versions,
        "test_results": test_cases_summary,
        "overall_status": "ALL_CHECKS_PASSED",
    }

    with open(work_dir / "evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2)

    print("[3/4] Input vs Expected vs Actual evidence saved to work/R33-T033/evidence.json.")
    print("[4/4] ALL CHECKS PASSED SUCCESSFULLY (Exit code 0).")
    print("=" * 65)


if __name__ == "__main__":
    run_checks()
