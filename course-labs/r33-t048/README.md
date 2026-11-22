# Course Lab R33-T048: Regression Error-Case Analysis

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T048 (Task 8 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T048`  
**Target Item:** `02_end_to_end_machine_learning_project.ipynb` (`## Experimenting with Attribute Combinations`)

---

## 1. Scope & Objective
Implements a testable error-analysis reporter on a deterministic synthetic regression fixture ($y = 2 x_1 + 1$). Confirms anomaly attribution when a deliberate prediction error is introduced on a held-out sample.

> ### ⚠️ Synthetic Pipeline Verification Notice
> The 6-row formula dataset (`r1..r6`) is a deterministic fixture used solely for metric and error reporting verification.

---

## 2. Invariants
- Injected error on `r6` is explicitly flagged under `OUTLIER_ANOMALY`.
- Clean records `r1..r5` are categorized as `EXACT_MATCH`.
- Summary category counts reconcile with detailed records.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t048 -p "test_*.py"
```
