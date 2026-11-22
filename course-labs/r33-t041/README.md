# Course Lab R33-T041: Time Series Data Contract Validation

**Student:** SHAKTI KUMAR SINGH  
**Task ID:** R33-T041 (Task 1 of 10)  
**Assigned Branch:** `student/e26b07f0780/R33-T041`  
**Target Item:** `15_processing_sequences_using_rnns_and_cnns.ipynb` (`# Basic RNNs`)

---

## 1. Scope & Objective
This component implements an explicit, CPU-only schema validator for time series observation records (`record_id`, `timestamp`, `value`). It verifies structural integrity and reports all field-level failures without silently dropping invalid rows.

> ### ⚠️ Synthetic Pipeline Verification Notice
> The 10-observation daily time series fixture in `fixture.json` and `fixture.csv` (`[10,12,11,13,15,14,16,18,17,19]`, IDs `d1..d10`) is a synthetic verification fixture designed solely for offline interface and contract testing. It does not represent real-world clinical or financial data.

---

## 2. Data Contract Specification

| Field | Type | Required | Constraint | Description |
|---|---|---|---|---|
| `record_id` | `str` / `int` | Yes | Non-empty, unique | Stable primary identifier for time-step |
| `timestamp` | `str` | Yes | YYYY-MM-DD ISO format | Calendar date of observation |
| `value` | `int` / `float` | Yes | Finite numeric scalar | Measured observation value |

### Chronological Partitioning:
- **Training Set (60%):** IDs `d1` to `d6` (values `[10, 12, 11, 13, 15, 14]`)
- **Validation Set (20%):** IDs `d7` and `d8` (values `[16, 18]`)
- **Test Set (20%):** IDs `d9` and `d10` (values `[17, 19]`)
- **No Shuffling:** Time order is strictly maintained.

---

## 3. Automated Verification Check
```bash
python3 -m unittest discover -s course-labs/r33-t041 -p "test_*.py"
```
