# Course Lab R33-T021: Deterministic Split Manifest & Leakage-Free Splitting

**Student:** Raghav Motwani (E26B07F1216)  
**Task ID:** R33-T021 (Task 1 of 10)  
**Assigned Repository:** `S-AI-Projects/handson-ml3`  
**Target Item:** `11_training_deep_neural_networks.ipynb` (Xavier and He Initialization)  
**Status:** Complete & Verified  

---

## 1. Goal & Data Contract
This module provides a deterministic, leakage-free split manifest builder that preserves group and temporal isolation across train, validation, and test splits.

- **Synthetic Fixture:** Six synthetic 4×4 grayscale images:
  - Three all-zero images labelled 0 (group IDs: `grp_alpha`, `grp_beta`).
  - Three images with a white diagonal (255) labelled 1 (group IDs: `grp_gamma`, `grp_delta`).
- **Data Contract:** Schema fields: `record_id`, `group_id`, `timestamp`, `pixels` (4x4 matrix), `label` (int 0 or 1).
- **Disclaimer:** Synthetic CPU smoke fixture for pipeline verification only. Not evidence of real-world or clinical performance.

---

## 2. Key Properties & Invariants
- **Disjoint Record Sets:** `train ∩ val == ∅`, `train ∩ test == ∅`, `val ∩ test == ∅`.
- **Group Isolation:** Records sharing a `group_id` are never separated across splits.
- **Chronological Ordering:** Temporal ordering is preserved in time-series splits (`train_time <= val_time <= test_time`).
- **Validation Failure:** Intentionally overlapping IDs or groups raise a descriptive `ValueError`.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t021 -p "test_*.py" -v
```
