# Course Lab R33-T025: Error Analysis

## Overview
Generates structured error-analysis reports comparing ground-truth centroid assignments against predictions on 2D synthetic fixture.

## Fixture & Error Injection
- Fixture: Six 2D points `(0,0)`, `(0,1)`, `(1,0)` -> cluster 0; `(5,5)`, `(5,6)`, `(6,5)` -> cluster 1.
- Injected error: Point `pt_003` prediction deliberately flipped from `0` to `1`.
- Error report records `record_id`, `expected`, `predicted`, and `reason_category`.

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t025 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
