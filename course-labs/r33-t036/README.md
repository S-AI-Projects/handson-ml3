# R33-T036: Add a Bounded CPU Smoke Run

**Task:** R33-T036 · 6 of 10  
**Student:** Snehakumari Vinod Singh  
**Section:** 06 decision trees / Sensitivity to axis orientation  
**Notebook:** `06_decision_trees.ipynb`  
**Branch:** `student/e26b07f0590/R33-T036`

---

## Goal

Build a tiny classification pipeline smoke test: at most 32 samples, a fixed seed, and at most one training epoch / 20 update steps. Use a small CPU model or deterministic baseline; assert interface correctness, not useful accuracy.

---

## Synthetic Fixture

> **⚠️ Disclaimer:** These records are **fully synthetic** — generated for pipeline smoke-test verification only. Results here are **not** evidence of real-world, clinical, or production performance and must not be cited as such.

Eight synthetic rows:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = 1.0` (constant)
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`

| record_id | group_id | x1 | x2 | label |
|-----------|----------|----|----|-------|
| s1        | g0       | 0  | 1  | 0     |
| s2        | g0       | 1  | 1  | 0     |
| s3        | g0       | 2  | 1  | 0     |
| s4        | g0       | 3  | 1  | 0     |
| s5        | g1       | 4  | 1  | 1     |
| s6        | g1       | 5  | 1  | 1     |
| s7        | g1       | 6  | 1  | 1     |
| s8        | g1       | 7  | 1  | 1     |

### Stratified Split Design
- **Train partition (6 rows):** `['s1', 's2', 's3', 's5', 's6', 's7']` (preserves both groups `g0`, `g1` and balanced classes)
- **Test partition (2 rows):** `['s4', 's8']` (isolated evaluation fixture)

---

## Smoke Run Constraints & Interface

- **Max samples:** ≤ 32 (enforced with `ValueError` on larger inputs)
- **Max steps:** ≤ 20 updates (enforced with `ValueError` on larger requested steps)
- **Max epochs:** ≤ 1 epoch
- **Hardware:** Pure CPU, standard-library only, zero external dependencies, no GPU or network access needed
- **Loss / outputs:** Finite float cross-entropy loss, valid probabilities in `[0.0, 1.0]`, valid discrete predictions `{0, 1}`
- **Baseline comparison:** Includes `NaiveConstantBaseline` explicitly labelled as majority vote

---

## Running Checks

```bash
python3 -m unittest discover -s course-labs/r33-t036 -p "test_*.py" -v
```

Run verification runner:
```bash
python3 work/R33-T036/run_verification.py
```

---

## Limits & Data Contract

- Standard library only — no NumPy, scikit-learn, PyTorch, or GPU acceleration required.
- Demonstrates pipeline boundary contract adherence and determinism under CPU execution.
- Not a performance or accuracy claim on any real benchmark.

## Optional Real-Data Extension

Any optional real-world extension must record:
- Source dataset name, URL, and license
- Access requirements and authentication steps
- Split strategy ensuring group/temporal isolation
- Hardware and software versions used
