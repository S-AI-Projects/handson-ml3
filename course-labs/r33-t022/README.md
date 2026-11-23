# Course Lab R33-T022: Honest Majority-Class Baseline

**Student:** Raghav Motwani (E26B07F1216)  
**Task ID:** R33-T022 (Task 2 of 10)  
**Assigned Repository:** `S-AI-Projects/handson-ml3`  
**Target Item:** `03_classification.ipynb` (Performance Measures)  
**Status:** Complete & Verified  

---

## 1. Goal & Data Contract
Provides the simplest honest classification baseline (majority class) fitted strictly on training data and evaluated on a fixed held-out fixture.

- **Synthetic Fixture:** Eight synthetic rows:
  - `x1`: `[0, 1, 2, 3, 4, 5, 6, 7]`
  - `x2`: `1`
  - `label`: `[0, 0, 0, 0, 1, 1, 1, 1]`
- **Data Contract:** Schema: `record_id`, `group_id`, `x1`, `x2`, `label`.
- **Disclaimer:** Synthetic CPU smoke fixture for pipeline verification only. No real-world accuracy claims are made.

---

## 2. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t022 -p "test_*.py" -v
```
