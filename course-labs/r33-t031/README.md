# Course Lab R33-T031: Image Data Contract Validation & tf.data Preprocessing

**Student:** Snehakumari Vinod Singh  
**Task Code:** R33-T031 (Task 1 of 10)  
**Assigned Branch:** `student/e26b07f0590/R33-T031`  
**Related Issue:** [Issue #31](https://github.com/S-AI-Projects/handson-ml3/issues/31)  
**Target Item:** `13_loading_and_preprocessing_data.ipynb` (The tf.data API)

---

## 1. Scope & Objective

This contribution provides a lightweight, self-contained, CPU-only data contract validator and pipeline component for image inputs within the `tf.data` pipeline ecosystem. It ensures that dataset records pass rigorous schema, dimension, typing, and domain constraints prior to ingest into training or evaluation pipelines.

> ### ⚠️ Synthetic Fixture Disclaimer
> The dataset fixture in `fixture.json` and `fixture.csv` contains six synthetic 4×4 grayscale image records designed strictly for CPU smoke testing and data contract verification.
>
> **These are NOT observations from any real-world image dataset.** They are simulated pipeline verification fixtures and must never be cited as evidence of real-world or clinical predictive performance.

---

## 2. Data Contract Specification

Each record in the dataset is validated against the following data contract:

| Field | Type | Domain / Shape | Description |
| :--- | :--- | :--- | :--- |
| `record_id` | `str` or `int` | Non-empty, unique across dataset | Unique sample record identifier. |
| `group_id` | `str` or `int` | Non-empty | Cohort, subject, or batch group identifier. |
| `pixels` | `List[List[float]]` / `List[List[int]]` | 4×4 2D matrix (or 16-element sequence), values in `[0, 255]` | Grayscale pixel matrix. Checks reject malformed tensors, non-numeric values, negative values, values > 255, and NaNs. |
| `label` | `int` | Binary `{0, 1}` | Class label: `0` for all-zero baseline images, `1` for white-diagonal images. |

---

## 3. Synthetic Fixture Details

The smoke fixture contains 6 synthetic records:
- **3 all-zero images (Label 0):** Every pixel is `0`.
- **3 white-diagonal images (Label 1):** Main diagonal pixels (`[i, i]`) are `255`, and off-diagonal pixels are `0`.
- **Groups:** Declared across cohorts (`grp_alpha`, `grp_beta`, `grp_gamma`).

---

## 4. Failure Handling & Non-Silent Dropping

Invalid records are never silently dropped or truncated. The `SchemaValidator` produces explicit `ValidationError` objects recording:
- `record_id`: The identifier of the failing record.
- `field`: The specific field that failed validation (`record_id`, `group_id`, `pixels`, `label`).
- `reason`: A human-readable diagnostic description of why validation failed.

---

## 5. Running Automated Checks & Tests

### Run Unit Tests
```bash
python3 -m unittest discover -s course-labs/r33-t031 -p "test_*.py" -v
```

### Run Verification Script (Inputs, Expected vs Actual, Evidence)
```bash
python3 work/R33-T031/run_verification.py
```
This script executes all unit tests, verifies assertions, checks tool versions, and saves reproducible evidence to `work/R33-T031/evidence.json`. It terminates with exit code `0` on success, or non-zero if any assertion fails.
