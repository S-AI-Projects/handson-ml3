# Course Lab R33-T032: Leakage-Free Splitting & Deterministic Split Manifest

**Student:** Snehakumari Vinod Singh  
**Task Code:** R33-T032 (Task 2 of 10)  
**Assigned Branch:** `student/e26b07f0590/R33-T032`  
**Related Issue:** [Issue #32](https://github.com/S-AI-Projects/handson-ml3/issues/32)  
**Target Item:** `06_decision_trees.ipynb` (Making Predictions)

---

## 1. Scope & Objective

This contribution implements a deterministic split manifest generator and leakage-prevention validation suite for decision tree classification workflows. It ensures that dataset records partitioned into training, validation, and test sets maintain strict mutual isolation, preventing data leakage across record identities, group/subject cohorts, and chronological time horizons.

> ### ⚠️ Synthetic Fixture Disclaimer
> The dataset fixture in `fixture.json` and `fixture.csv` contains eight synthetic classification rows designed strictly for CPU smoke testing and split pipeline verification.
>
> **These are NOT observations from any named real-world dataset.** They are synthetic verification fixtures and must never be cited as evidence of real-world or clinical predictive performance.

---

## 2. Synthetic Fixture Specification

The 8-row classification smoke fixture follows this exact data contract:

| Field | Type | Domain / Values | Description |
| :--- | :--- | :--- | :--- |
| `record_id` | `str` | `rec_001` through `rec_008` (unique) | Stable sample identifier. |
| `group_id` | `str` | `grp_001` through `grp_004` | Subject/cohort grouping identifier. |
| `x1` | `int` | `[0, 1, 2, 3, 4, 5, 6, 7]` | Primary numeric feature. |
| `x2` | `int` | `[1, 1, 1, 1, 1, 1, 1, 1]` | Secondary constant feature. |
| `label` | `int` | `[0, 0, 0, 0, 1, 1, 1, 1]` | Binary classification target. |
| `timestamp` | `int` | `[100, 101, ..., 107]` | Monotonic sequence for temporal split testing. |

---

## 3. Deterministic Split Manifest

The split manifest (`split_manifest.json`) records the exact record IDs allocated to each partition:

- **Training Partition (4 records):** `rec_001`, `rec_002` (class 0); `rec_005`, `rec_006` (class 1)
- **Validation Partition (2 records):** `rec_003` (class 0); `rec_007` (class 1)
- **Test Partition (2 records):** `rec_004` (class 0); `rec_008` (class 1)

### Leakage Prevention Guarantees:
1. **Disjoint Sets:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Stratification Balance:** Exact 50% class distribution preserved across all splits.
3. **Repeatable Determinism:** Repeated invocations produce identical manifests across runs.
4. **Leakage Rejection:** Overlapping record IDs, group leakage, and forward time-series lookahead violations are actively detected and rejected.

---

## 4. Running Automated Checks & Tests

### Run Unit Tests
```bash
python3 -m unittest discover -s course-labs/r33-t032 -p "test_*.py" -v
```

### Run Verification Runner (Tool versions, evidence, exit code 0)
```bash
python3 work/R33-T032/run_verification.py
```
Outputs:
- Generates `work/R33-T032/tool_versions.json`
- Generates `work/R33-T032/unittest_output.txt`
- Generates `work/R33-T032/evidence.json`
- Exits with code `0` on success or non-zero on assertion failure.
