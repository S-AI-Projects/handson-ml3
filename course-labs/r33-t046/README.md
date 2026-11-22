# Course Lab R33-T046: Decision Tree Configuration Schema

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T046 (Task 6 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T046`  
**Target Item:** `06_decision_trees.ipynb` (`# Decision Trees Have High Variance`)

---

## 1. Scope & Objective
Implements a strict serializable configuration schema (`seed`, `schema_version`, `feature_ordering`, `split_ids`, `algorithm_settings`) that ensures reproducibility and rejects unknown or contradictory parameters.

> ### ⚠️ Synthetic Pipeline Verification Notice
> The 8-row synthetic dataset ($x_1 \in [0..7], x_2 = 1, y \in \{0, 1\}$) is a deterministic fixture for schema testing.

---

## 2. Configuration Invariants
- Fixed stratified split: 4 training (2 positive, 2 negative), 2 validation, 2 test.
- Rejection of unknown keys, invalid criteria, negative depths, and overlapping split partitions.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t046 -p "test_*.py"
```
