# Course Lab R33-T002: Deterministic Split Manifest & Leakage-Free Splitting

**Student:** NAKSHATRA MESHRAM  
**Task Code:** R33-T002 (Task 2 of 10)  
**Assigned Branch:** `student/e26b07f0682` / `R33-T002`  
**Target Item:** `04_training_linear_models.ipynb` (Section: The Normal Equation)

---

## 1. Scope & Objective

This contribution provides an offline, leakage-free deterministic split manifest for regression datasets in linear modeling pipelines (The Normal Equation). It guarantees:
1. Strict disjointness between train, validation, and test partitions ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
2. Complete group and subject isolation (preventing patient/user ID leakage across partitions).
3. Chronological time-series ordering without temporal lookahead leakage ($\max(\text{train.ts}) < \min(\text{val.ts})$ and $\max(\text{val.ts}) < \min(\text{test.ts})$).
4. Fully reproducible, deterministic manifest generation.

> ### ⚠️ Synthetic Formula Fixture Disclaimer
> The 6-row regression dataset in `fixture.json` and `fixture.csv` is generated deterministically from the linear formula:
> $$\text{target} = 2 \cdot x_1 + 1, \quad x_2 = 1, \quad x_1 \in \{0, 1, 2, 3, 4, 5\}$$
> 
> **These are NOT observations from any real-world dataset.** They serve solely as a deterministic fixture to verify data isolation, split contracts, and pipeline integrity.

---

## 2. Data Contract & Schema

Each record contains:

| Field | Type | Required | Description |
|---|---|---|---|
| `record_id` | `str` / `int` | Yes | Unique row identifier (`r1` through `r6`). |
| `x1` | `int` / `float` | Yes | Independent feature 1 ($x_1 \in [0, 5]$). |
| `x2` | `int` / `float` | Yes | Intercept or feature 2 ($x_2 = 1$). |
| `target` | `int` / `float` | Yes | Continuous regression target ($2 \cdot x_1 + 1$). |
| `timestamp` | `int` / `float` | Optional | Sequence or time index for temporal ordering verification. |
| `group_id` | `str` | Optional | Group/cluster identifier for group isolation checks. |

---

## 3. Split Manifest Specification

The default deterministic formula split assigns:

- **Training Set (4 rows):** `["r1", "r2", "r3", "r4"]` ($x_1 \in [0, 3]$, $\text{target} \in [1, 7]$)
- **Validation Set (1 row):** `["r5"]` ($x_1 = 4$, $\text{target} = 9$)
- **Test Set (1 row):** `["r6"]` ($x_1 = 5$, $\text{target} = 11$)

### Manifest Structure
```json
{
  "manifest_version": "1.0",
  "split_strategy": "deterministic_formula",
  "train_record_ids": ["r1", "r2", "r3", "r4"],
  "validation_record_ids": ["r5"],
  "test_record_ids": ["r6"],
  "counts": {
    "train": 4,
    "validation": 1,
    "test": 1,
    "total": 6
  },
  "is_disjoint": true
}
```

---

## 4. Automated Verification & Testing

Execute the test suite with standard library `unittest` (offline, CPU-only):

```bash
python3 -m unittest discover -s course-labs/r33-t002 -p "test_*.py" -v
```

The test suite asserts:
1. Exact row count ($6$) and adherence to the deterministic equation $\text{target} = 2 \cdot x_1 + 1$.
2. Clean passage and disjoint manifest generation for both JSON and CSV fixtures.
3. Strict rejection of overlapping record IDs across any partition pair.
4. Strict rejection of group leakage across partition boundaries.
5. Strict rejection of chronological inversion in temporal splits.
6. Bitwise reproducibility across repeated runs.

---

## 5. Real-Data Extension Guidance

If adapting this component to real-world regression benchmarks (such as the California Housing dataset):
- **Source:** Scikit-Learn `fetch_california_housing()` or StatLib repository.
- **Licence:** Public domain / CC0.
- **Split Design:** Spatial block splitting or cluster-based group K-fold to avoid spatial autocorrelation leakage between neighboring districts.
