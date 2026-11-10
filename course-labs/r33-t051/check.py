import json, platform, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent / "work" / "R33-T051"
sys.path.insert(0, str(HERE))

import fixture
from validator import validate_records


def main():
    WORK.mkdir(parents=True, exist_ok=True)

    expected = {"valid_fixture_failures": 0, "min_test_count": 1}
    actual_failures = validate_records(fixture.RECORDS)

    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py", top_level_dir=str(HERE))
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    (WORK / "inputs.json").write_text(json.dumps(fixture.RECORDS, indent=2))
    (WORK / "expected.json").write_text(json.dumps(expected, indent=2))
    (WORK / "actual.json").write_text(json.dumps({
        "valid_fixture_failures": len(actual_failures),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
    }, indent=2))
    (WORK / "versions.json").write_text(json.dumps({
        "python": platform.python_version(),
        "platform": platform.platform(),
    }, indent=2))
    (WORK / "NOTE.txt").write_text(
        "Synthetic fixture: this is PIPELINE VERIFICATION only, "
        "not evidence of real-world or clinical performance.\n")

    ok = (result.wasSuccessful()
          and result.testsRun >= expected["min_test_count"]
          and len(actual_failures) == expected["valid_fixture_failures"])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
