# Course Lab R33-T050: Bounded CPU Image Pipeline Smoke Test

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T050 (Task 10 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T050`  
**Target Item:** `13_loading_and_preprocessing_data.ipynb` (`## Putting Everything Together + Prefetching`)

---

## 1. Scope & Objective
Executes a lightweight, CPU-bounded training/inference pipeline smoke test ($\le 32$ samples, 1 epoch, $\le 20$ gradient updates) to assert interface contract correctness and finite losses without GPU or external credentials.

> ### ⚠️ Synthetic Pipeline Verification Notice
> This CPU smoke test utilizes 6 synthetic 4x4 image samples purely to verify optimization and pipeline execution contracts.

---

## 2. Invariants
- Loss values are strictly finite (non-NaN, non-Inf).
- Update step count is bounded ($\le 20$).
- Deterministic behavior guaranteed by seed control.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t050 -p "test_*.py"
```
