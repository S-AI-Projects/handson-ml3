"""Offline, CPU-only, bounded check for R33-T060.

Runs the unittest suite, requires a NONZERO test count, writes evidence to
work/R33-T060/, and exits non-zero on any failure.  It also runs the suite
with a deliberately changed expected result, which must FAIL.
"""
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVID = os.path.join(ROOT, "work", "R33-T060")


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
    import fixture
    from preprocess import ContaminationError, PreprocessError, Scaler, serving_request

    rc, n, out = run_suite()
    mrc, mn, mout = run_suite({"R33_T060_MUTATE_EXPECTED": "1"})

    s = Scaler()
    s.fit([r for r in fixture.FIXTURE if fixture.split_of(r) == "train"])
    outputs = {r["record_id"]: s.transform(fixture.to_grid(r["pixels"])) for r in fixture.FIXTURE}

    def outcome(fn):
        try:
            fn()
            return "accepted"
        except ContaminationError as e:
            return "rejected (ContaminationError): %s" % e
        except PreprocessError as e:
            return "rejected (PreprocessError): %s" % e

    grid = fixture.to_grid(fixture.FIXTURE[1]["pixels"])
    bad3 = [row[:3] for row in grid[:3]]
    bad_range = [list(r) for r in grid]
    bad_range[1][2] = 256

    evidence = {
        "task": "R33-T060",
        "label": "PIPELINE VERIFICATION on a synthetic CPU smoke fixture; not evidence of real-world or clinical performance",
        "inputs": {"fixture": fixture.FIXTURE, "splits": fixture.SPLITS},
        "expected": {"dtype": "float32", "shape": [4, 4, 1], "range": [0.0, 1.0],
                     "zero_image": "all 0.0", "diagonal_image": "1.0 on diagonal, else 0.0"},
        "actual": {rid: {"dtype": o.dtype, "shape": list(o.shape),
                         "min": min(o.flat), "max": max(o.flat), "flat": o.flat}
                   for rid, o in outputs.items()},
        "failure_cases": {
            "3x3 input": outcome(lambda: s.transform(bad3)),
            "pixel 256": outcome(lambda: s.transform(bad_range)),
            "contamination (fit on test record r4)": outcome(lambda: Scaler().fit([fixture.FIXTURE[4]])),
            "contamination (fit on whole fixture)": outcome(lambda: Scaler().fit(fixture.FIXTURE)),
        },
        "tf_serving_request_example_first_instance_shape": [4, 4, 1],
        "tf_serving_request_sent": False,
        "serving_body_bytes": len(serving_request(list(outputs.values()))),
        "suite": {"tests_run": n, "exit_code": rc},
        "mutation_check": {"tests_run": mn, "exit_code": mrc, "failed_as_required": mrc != 0},
        "tool_versions": {"python": platform.python_version(), "platform": platform.platform(),
                          "tensorflow": "not used (stdlib only)"},
    }
    with open(os.path.join(EVID, "evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=2)
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
