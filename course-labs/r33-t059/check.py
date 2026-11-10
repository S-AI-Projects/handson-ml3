"""Offline, CPU-only, bounded check for R33-T059.

Runs the unittest suite, requires a NONZERO test count, writes evidence to
work/R33-T059/, and exits non-zero on any failure.  It also runs the suite
with a deliberately changed expected result, which must FAIL.
"""
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVID = os.path.join(ROOT, "work", "R33-T059")


def run_suite(env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    p = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", HERE, "-p", "test_*.py", "-v"],
        capture_output=True, text=True, env=env, timeout=60, cwd=ROOT,
    )
    out = p.stdout + p.stderr
    n = 0
    for line in out.splitlines():
        if line.startswith("Ran "):
            n = int(line.split()[1])
    return p.returncode, n, out


def main():
    os.makedirs(EVID, exist_ok=True)
    sys.path.insert(0, HERE)
    import fixture, pipeline

    rc, n, out = run_suite()
    mrc, mn, mout = run_suite({"R33_T059_MUTATE_EXPECTED": "1"})

    summary = pipeline.summarise([pipeline.to_example(r) for r in fixture.FIXTURE])
    shuf = pipeline.shuffled(fixture.FIXTURE, 59)
    shuf_summary = pipeline.summarise([pipeline.to_example(r) for r in shuf])

    evidence = {
        "task": "R33-T059",
        "label": "PIPELINE VERIFICATION on a synthetic CPU smoke fixture; not evidence of real-world or clinical performance",
        "inputs": {
            "fixture": fixture.FIXTURE,
            "shuffled_order_seed59": [r["record_id"] for r in shuf],
        },
        "expected": {"predictions": summary["predictions"], "accuracy": 1.0,
                     "class_counts": {"0": 3, "1": 3}},
        "actual_original_order": summary,
        "actual_shuffled_after_id_alignment": shuf_summary,
        "order_invariant_after_alignment": summary == shuf_summary,
        "unsupported_inputs": {
            "missing_field": "rejected (SchemaError)",
            "shape_mismatch": "rejected (SchemaError)",
            "unseen_label": "mapped to fallback label -1, excluded from accuracy",
        },
        "suite": {"tests_run": n, "exit_code": rc},
        "mutation_check": {"tests_run": mn, "exit_code": mrc, "failed_as_required": mrc != 0},
        "tool_versions": {"python": platform.python_version(), "platform": platform.platform(),
                          "tensorflow": "not used (stdlib TFRecord codec)"},
    }
    with open(os.path.join(EVID, "evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=2, default=str)
    with open(os.path.join(EVID, "test_output.txt"), "w") as fh:
        fh.write("== normal run ==\n" + out + "\n== mutated expectation (must fail) ==\n" + mout)

    print("tests run: %d, exit code: %d" % (n, rc))
    print("mutation run: %d tests, exit code %d (must be non-zero)" % (mn, mrc))
    if n == 0 or rc != 0 or mrc == 0:
        print("CHECK FAILED")
        return 1
    print("CHECK OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
