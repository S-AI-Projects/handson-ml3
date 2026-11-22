# R33-T035: Make preprocessing explicit

**Task:** R33-T035 · 5 of 10  
**Student:** Snehakumari Vinod Singh  
**Section:** 09 unsupervised learning / Multiple Initializations  
**Notebook:** `09_unsupervised_learning.ipynb`  
**Branch:** `student/e26b07f0590/R33-T035`

---

## Goal

Fit any scaling or projection on training data only and retain the fitted parameters so the same transform can be applied to held-out data without leaking information from those rows.

This implements a `MinMaxScaler` (standard-library only) that fits exclusively on training rows and a nearest-centroid classifier using Euclidean distance.

---

## Synthetic Fixture

> **⚠️ Disclaimer:** These records are **fully synthetic** — generated for pipeline verification only. Results here are **not** evidence of real-world, clinical, or production performance and must not be cited as such.

| record_id | x1 | x2 |
|-----------|----|----|
| r1        | 0  | 0  |
| r2        | 0  | 1  |
| r3        | 1  | 0  |
| r4        | 5  | 5  |
| r5        | 5  | 6  |
| r6        | 6  | 5  |

**Fixed centroids:** c0 = (0, 0), c1 = (5, 5)  
**Expected nearest-centroid assignments:** `[0, 0, 0, 1, 1, 1]`

---

## Data Contract

| Field              | Value |
|--------------------|-------|
| Source             | Fully synthetic |
| Licence            | N/A |
| Real dataset used  | No |
| Schema             | record_id (str), x1 (float), x2 (float) |
| Fixture files      | `fixture.csv`, `fixture.json` |

---

## Implementation

### `pipeline.py`
- **`MinMaxScaler`** — fits on training data, retains `_min`, `_max`, `_range`; `transform()` raises `RuntimeError` if called before `fit()`; constant-feature raises `ValueError`
- **`nearest_centroid_assignments(points, centroids)`** — assigns each point to the nearest centroid by Euclidean distance
- **`run_pipeline(train_rows, test_rows, centroids)`** — full pipeline: fit scaler on train, scale train+test+centroids with same params, return assignments

### Key design decisions

1. **No re-fit on test data** — `transform()` is called once fitted; it has no internal `fit()` path.
2. **Constant-feature rejection** — zero-range columns raise `ValueError` at `fit()` time.
3. **Deterministic** — no random state; centroid assignment is a pure deterministic function.

---

## Running the Tests

```bash
python3 -m unittest discover -s course-labs/r33-t035 -p "test_*.py"
```

**Expected output:** 15 tests, all pass, exit code 0.

---

## Verification

```bash
python3 work/R33-T035/run_verification.py
```

Saves `evidence.json`, `tool_versions.json`, `unittest_output.txt` in `work/R33-T035/`. Exits non-zero on assertion failure.

---

## Limits

- Standard library only — no NumPy, scikit-learn, or external packages.
- Operates on the six-row synthetic fixture; not validated on real clustering benchmarks.
- MinMaxScaler does not handle infinite or NaN values.

## Optional Real-Data Extension

Any extension to real data must separately document:
- Dataset name, source URL, and licence
- Access requirements (download, sign-up, etc.)
- Split design (train/validation/test sizes, stratification strategy)
- Exact package versions used
