#!/usr/bin/env python3
import subprocess, sys

cmd = ["python3", "-m", "unittest", "discover", "-s", "course-labs/r33-t044", "-p", "test_*.py", "-v"]
print("Running verification check:", " ".join(cmd))
res = subprocess.run(cmd)
if res.returncode != 0:
    print("Verification check failed!", file=sys.stderr)
    sys.exit(res.returncode)
print("Verification check passed successfully!")
