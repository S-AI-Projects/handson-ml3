import json
import platform
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent / "work" / "R33-T052"
sys.path.insert(0, str(HERE))

import fixture
from assign import nearest_centroid
from manifest import make_manifest

EXPECTED = {
    "assignments": [0, 0, 0, 1, 1, 1],
    "manifest_splits": {
        "train": ["p2", "p3", "p4", "p5"],
        "validation": ["p6"],
        "test": ["p1"],
    },
    "min_test_count": 1,
}


def main():
    WORK.mkdir(parents=True, exist_ok=True)

    assignments = nearest_centroid(fixture.RECORDS, fixture.CENTROIDS)
    manifest = make_manifest(fixture.RECORDS, seed=fixture.SEED)

    suite = unittest.defaultTestLoader.discover(
        str(HERE), pattern="test_*.py", top_level_dir=str(HERE)
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    actual = {
        "assignments": assignments,
        "manifest_splits": manifest["splits"],
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }
    inputs = {
        "records": fixture.RECORDS,
        "centroids": fixture.CENTROIDS,
        "seed": fixture.SEED,
    }
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    note = (
        "Synthetic fixture: this is PIPELINE VERIFICATION only, "
        "not evidence of real-world or clinical performance.\n"
    )

    (WORK / "inputs.json").write_text(json.dumps(inputs, indent=2))
    (WORK / "expected.json").write_text(json.dumps(EXPECTED, indent=2))
    (WORK / "actual.json").write_text(json.dumps(actual, indent=2))
    (WORK / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    (WORK / "versions.json").write_text(json.dumps(versions, indent=2))
    (WORK / "NOTE.txt").write_text(note)

    ok = (
        result.wasSuccessful()
        and result.testsRun >= EXPECTED["min_test_count"]
        and actual["assignments"] == EXPECTED["assignments"]
        and actual["manifest_splits"] == EXPECTED["manifest_splits"]
    )
    print("check:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
