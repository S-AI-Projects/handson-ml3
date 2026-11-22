# Course Lab R33-T049: Input Perturbation Robustness

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T049 (Task 9 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T049`  
**Target Item:** `12_custom_models_and_training_with_tensorflow.ipynb` (`### Type Conversions`)

---

## 1. Scope & Objective
Verifies input transformation robustness: ordering invariance across sample permutations when aligned by stable `record_id`, and explicit rejection of malformed dimensions and missing fields.

> ### ⚠️ Synthetic Pipeline Verification Notice
> The synthetic 4x4 image fixture validates interface invariants, not adversarial defense or general robustness against image perturbations.

---

## 2. Invariants
- Output predictions indexed by `record_id` are identical under arbitrary permutations of the input list.
- Shape mismatches (e.g. $3\times 3$) and missing required keys raise explicit exceptions.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t049 -p "test_*.py"
```
