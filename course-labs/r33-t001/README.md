# Course Lab R33-T001: Data Contract Validation & Schema Enforcement

**Student:** NAKSHATRA MESHRAM  
**Task Code:** R33-T001 (Task 1 of 10)  
**Assigned Branch:** `student/e26b07f0682` / `R33-T001`  
**Target Item:** `03_classification.ipynb` (Classification / MNIST input pipeline)

---

## 1. Scope & Objective

This contribution implements a self-contained, CPU-only schema validator and contract test suite for classification input records. It ensures that data passed to classification pipelines satisfies expected structure, typing, uniqueness, and completeness constraints before training or evaluation.

> ### ⚠️ Synthetic Data Disclaimer
> The 8-row dataset fixture in `fixture.json` and `fixture.csv` contains purely synthetic numerical values and mock identifiers designed solely for pipeline verification and continuous integration checks.
> 
> **These are NOT observations from the real MNIST dataset.** They must not be used as evidence of real-world or clinical predictive performance.

---

## 2. Data Contract Specification

Each record in the dataset must conform to the following schema:

| Field | Type | Required | Constraint | Description |
|---|---|---|---|---|
| `record_id` | `str` / `int` | Yes | Non-empty, unique | Stable primary identifier for the sample. |
| `group_id` | `str` / `int` | Yes | Non-empty | Stable group or cohort identifier. |
| `x1` | `int` / `float` | Yes | Non-NaN, finite scalar | Numerical input feature 1 ($x_1 \in [0, 7]$ in fixture). |
| `x2` | `int` / `float` | Yes | Non-NaN, finite scalar | Numerical input feature 2 ($x_2 = 1$ in fixture). |
| `label` | `int` | Yes | $\in \{0, 1\}$ | Binary classification target class. |

### Contract Rules & Failure Reporting:
- **No Silent Dropping:** Every record failure is reported with `record_id`, failing `field`, and descriptive `reason`.
- **Duplicate Detection:** Duplicate `record_id` entries across the input stream are flagged.
- **Type Safety:** Rejects non-numeric text, lists/tensors, boolean scalars, `NaN`, and infinite values for features `x1` and `x2`.
- **Target Integrity:** Rejects non-binary or floating-point class labels.

---

## 3. Synthetic Verification Fixture

The literal fixture contains 8 synthetic rows:
- `x1 = [0, 1, 2, 3, 4, 5, 6, 7]`
- `x2 = [1, 1, 1, 1, 1, 1, 1, 1]`
- `label = [0, 0, 0, 0, 1, 1, 1, 1]`

Available in:
- `course-labs/r33-t001/fixture.json`
- `course-labs/r33-t001/fixture.csv`

---

## 4. Stratified Split Design

To ensure reproducible, deterministic train/validation/test partitions on small verification fixtures:
- The dataset is stratified strictly by `label` ($50\%$ train, $50\%$ test).
- With 4 negative (`label=0`) and 4 positive (`label=1`) instances, both partitions receive exactly 2 negative and 2 positive instances.
- Implementation: `fixed_stratified_split(records, train_ratio=0.5)` in `pipeline.py`.

---

## 5. Automated Verification & Testing

Run the test suite using standard library `unittest` (offline, CPU-only):

```bash
python3 -m unittest discover -s course-labs/r33-t001 -p "test_*.py" -v
```

All 9 test cases verify:
1. Exact fixture dimensions, types, and values.
2. Clean passage of valid JSON and CSV fixtures.
3. Explicit rejection of missing or empty `record_id`.
4. Rejection of duplicate `record_id`.
5. Rejection of missing `group_id`.
6. Rejection of strings, tensors, `NaN`, and booleans in numeric fields.
7. Rejection of out-of-range or non-integer labels.
8. Preservation of stratified class balance across splits.

---

## 6. Limits & Optional Real-Data Extension Guidance

- **Limitations:** This component strictly validates tabular 2D feature vectors and binary targets. Complex 28x28 image tensors (such as raw MNIST pixel arrays) require a downstream tensor validator to verify shape `(28, 28)` or flattened shape `(784,)`.
- **Real-Data Extension:** If integrating real MNIST data:
  - Source: OpenML (`fetch_openml('mnist_784', as_frame=False)`) or Yann LeCun / Corinna Cortes MNIST archive.
  - Licence: Creative Commons Attribution-Share Alike 3.0.
  - Access Requirements: Publicly accessible network fetch.
  - Split Design: Standard 60,000 train / 10,000 test partition.
