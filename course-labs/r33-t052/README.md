# R33-T052: Deterministic split manifest for a K-Means fixture

Task: write a deterministic train/validation/test split manifest for the
"09 unsupervised learning / K-Means" section. This is the bounded component
only, not a reproduction of the full project.

## Data contract
- Record schema: `record_id` (unique non-empty string), `x1`, `x2` (numbers).
- Optional isolation fields: `group`, `user`, `time`.
- Records sharing a `group` or `user` (directly or through a chain) always
  land in the same split.
- If every record has `time`, the split is chronological: train is earliest,
  then validation, then test.

## Data
Six **synthetic** fictional points with IDs p1..p6:
(0,0), (0,1), (1,0), (5,5), (5,6), (6,5). This is a pipeline fixture, not a
real dataset. Results are pipeline verification only, not evidence of
real-world or clinical performance.

## Result
- For fixed centroids (0,0) and (5,5), the nearest-centroid assignments are
  0,0,0,1,1,1.
- Manifest for seed 0 (also saved in `split_manifest.json`):
  train p2, p3, p4, p5; validation p6; test p1.
- Splits are disjoint and cover every record. Repeated runs, and a different
  input order, give the same manifest (units are ordered by SHA-256 of
  seed and unit key, not by Python's random module).
- Overlapping IDs, overlapping groups or users, and non-chronological
  time splits are rejected with a specific reason.

## Limits
Structure and split logic only; no clustering is trained. The fixture itself
has no group, user or time fields, so those rules are tested on extra
synthetic records inside the tests. Offline, CPU-only, stdlib only.

## Optional real-data extension
Any real dataset must separately record its source, licence, access
requirements and split design.

## Run
    python3 course-labs/r33-t052/manifest.py
    python3 -m unittest discover -s course-labs/r33-t052 -p "test_*.py"
    python3 course-labs/r33-t052/check.py
