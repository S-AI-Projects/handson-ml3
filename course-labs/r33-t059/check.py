"""Runs the unit tests, saves evidence, then runs a deliberately changed
expectation that must fail (exit code 1)."""
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORK = os.path.join(ROOT, "work", "R33-T059")
sys.path.insert(0, HERE)
import pipeline as p  # noqa: E402


def main():
    os.makedirs(WORK, exist_ok=True)
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", HERE, "-p", "test_*.py"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    log = proc.stderr + proc.stdout
    with open(os.path.join(WORK, "unittest_output.txt"), "w") as f:
        f.write(log)
    ran = [ln for ln in log.splitlines() if ln.startswith("Ran ")]
    count = int(ran[0].split()[1]) if ran else 0
    if proc.returncode != 0 or count == 0:
        print("unit tests failed or zero tests ran")
        return 1

    fixture = p.make_fixture()
    base = p.evaluate(fixture)
    shuf = p.evaluate(p.shuffled(fixture))
    expected = {"accuracy": 1.0, "scored": 6}
    actual = {"accuracy": base["accuracy"], "scored": base["scored"]}
    evidence = {
        "seed": p.SEED, "tests_run": count,
        "inputs": [r["record_id"] for r in fixture],
        "shuffled_order": [r["record_id"] for r in p.shuffled(fixture)],
        "expected": expected, "actual": actual,
        "order_invariant": base == shuf,
        "versions": {"python": platform.python_version(),
                     "platform": platform.platform()},
    }
    with open(os.path.join(WORK, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)
    assert actual == expected and base == shuf

    # Deliberately changed expectation: must fail and exit non-zero.
    wrong = {"accuracy": 0.5, "scored": 6}
    if actual != wrong:
        print(f"deliberate failure: expected {wrong}, got {actual}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
