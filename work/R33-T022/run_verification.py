#!/usr/bin/env python3
import subprocess, sys
cmd = ["python3", "-m", "unittest", "discover", "-s", "course-labs/r33-t022", "-p", "test_*.py", "-v"]
res = subprocess.run(cmd)
sys.exit(res.returncode)
