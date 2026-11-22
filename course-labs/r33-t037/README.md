# R33-T037: Record Reproducible Configuration

**Task:** R33-T037 · 7 of 10  
**Student:** Snehakumari Vinod Singh  
**Section:** 15 processing sequences using rnns and cnns / LSTMs  
**Notebook:** `15_processing_sequences_using_rnns_and_cnns.ipynb`  
**Branch:** `student/e26b07f0590/R33-T037`

---

## Goal

Write a serializable configuration for the time series fixture with:
- Random seed
- Schema version
- Feature ordering
- Explicit split IDs
- Algorithm settings

Strictly reject unknown fields or contradictory settings (e.g., overlapping splits, incompatible bounds, missing features).

---

## Synthetic Fixture

> **⚠️ Disclaimer:** These records are **fully synthetic** — generated for pipeline configuration reproducibility verification only. Results here are **not** evidence of real-world, clinical, or production performance and must not be cited as such.

Ten consecutive daily observations with values `[10, 12, 11, 13, 15, 14, 16, 18, 17, 19]` and IDs `d1..d10`.

| record_id | timestamp  | value | Split |
|-----------|------------|-------|-------|
| d1        | 2026-01-01 | 10.0  | Train |
| d2        | 2026-01-02 | 12.0  | Train |
| d3        | 2026-01-03 | 11.0  | Train |
| d4        | 2026-01-04 | 13.0  | Train |
| d5        | 2026-01-05 | 15.0  | Train |
| d6        | 2026-01-06 | 14.0  | Train |
| d7        | 2026-01-07 | 16.0  | Validation |
| d8        | 2026-01-08 | 18.0  | Validation |
| d9        | 2026-01-09 | 17.0  | Test |
| d10       | 2026-01-10 | 19.0  | Test |

- **Sequential Partition:** First 6 training, next 2 validation, final 2 test (no shuffling).

---

## Configuration Schema & Validation

```json
{
  "schema_version": "1.0.0",
  "seed": 42,
  "feature_ordering": ["value"],
  "split_ids": {
    "train": ["d1", "d2", "d3", "d4", "d5", "d6"],
    "validation": ["d7", "d8"],
    "test": ["d9", "d10"]
  },
  "algorithm_settings": {
    "model_type": "autoregressive_lag",
    "lag": 1,
    "clip_min": 0.0,
    "clip_max": 100.0,
    "learning_rate": 0.01
  }
}
```

### Strict Rejection Rules
1. **Unknown Keys:** Rejects unknown top-level or algorithm keys with `ValueError`.
2. **Contradictory Splits:** Rejects any partition overlap between train, validation, or test sets.
3. **Invalid Algorithm Parameters:** Validates `lag < train_size`, `clip_min < clip_max`, positive seed.
4. **Input Shape & Types:** Rejects missing `record_id`, missing features from `feature_ordering`, or non-finite values.

---

## Running Checks

Run unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t037 -p "test_*.py" -v
```

Run verification runner:
```bash
python3 work/R33-T037/run_verification.py
```

---

## Limits & Data Contract

- Standard library only — no external deep learning or NLP packages required.
- Synthetic fixture for sequence verification only; not indicative of real-world sequence modeling capabilities.

## Optional Real-Data Extension

Any optional real-world extension must record:
- Source sequence dataset name, URL, and license
- Temporal partitioning timestamp cutoff
- Exact software and framework versions used
