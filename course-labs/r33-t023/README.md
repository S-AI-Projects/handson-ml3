# Course Lab R33-T023: Preprocessing Contract

## Overview
Implements a strict preprocessing contract for 2D synthetic point fixtures, retaining fitted scaling parameters and preventing leakage from held-out data.

## Fixture Specification
Six 2D points with stable IDs:
- `(0, 0)`, `(0, 1)`, `(1, 0)` -> nearest centroid `(0, 0)` -> cluster `0`
- `(5, 5)`, `(5, 6)`, `(6, 5)` -> nearest centroid `(5, 5)` -> cluster `1`
- Nearest centroid assignments: `[0, 0, 0, 1, 1, 1]`.

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t023 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
