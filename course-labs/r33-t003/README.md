# Course Lab R33-T003: Minimal Majority-Class Baseline for Image Classification

**Student:** NAKSHATRA MESHRAM  
**Task Code:** R33-T003 (Task 3 of 10)  
**Assigned Branch:** `student/e26b07f0682` / `R33-T003`  
**Target Item:** `19_training_and_deploying_at_scale.ipynb` (Section: Exporting SavedModels)

---

## 1. Scope & Objective

This contribution provides the simplest honest baseline (`MajorityClassBaseline`) for image classification under `19_training_and_deploying_at_scale.ipynb`. It provides a zero-leakage, un-fitted comparison benchmark:
1. Learns the majority class using **only training data**.
2. Predicts the constant majority class on unseen held-out samples.
3. Computes objective baseline classification metrics (accuracy, class distribution) on the test split.
4. Validates the 4x4 grayscale image contract (`record_id`, `group_id`, `pixels`, `label`).

> ### ⚠️ Synthetic Smoke Fixture Disclaimer
> The 6-row dataset in `fixture.json` and `fixture.csv` consists of synthetic 4×4 pixel arrays (16 elements):
> - **Three all-zero images** (`pixels = [0]*16`, `label = 0`)
> - **Three diagonal images** (pixels with value 255 on main diagonal $(0, 5, 10, 15)$, `label = 1`)
> 
> **These are CPU smoke verification fixtures, NOT the named real dataset.** They must not be cited as evidence of real-world or clinical predictive performance.

---

## 2. Image Data Contract & Schema

| Field | Type | Description |
|---|---|---|
| `record_id` | `str` / `int` | Unique sample identifier (`img_01` .. `img_06`). |
| `group_id` | `str` | Cohort / generator group identifier (`grp_zero_1`, `grp_diag_1`, etc.). |
| `pixels` | `List[int]` | 16 integer grayscale pixel values in range $[0, 255]$. |
| `label` | `int` | Binary class target ($\in \{0, 1\}$). |

---

## 3. Baseline Model & Partition Evaluation

### Default Benchmark Partition:
- **Training Set (4 samples):** `img_01`, `img_02`, `img_03` (three 0s), and `img_04` (one 1).
  - Training class distribution: `{0: 3, 1: 1}`.
  - Learned majority class: `0`.
- **Held-out Test Set (2 samples):** `img_05`, `img_06` (both label 1).
  - Predictions on test set: `[0, 0]`.
  - Correct predictions: `0 / 2`.
  - Baseline Test Accuracy: **`0.0%`**.

This provides an honest, empirical lower bound that any trained deep learning model (e.g., exported SavedModel) must surpass.

---

## 4. Automated Verification & Testing

Execute the test suite using standard library `unittest` (offline, CPU-only):

```bash
python3 -m unittest discover -s course-labs/r33-t003 -p "test_*.py" -v
```

The test suite asserts:
1. Exact fixture dimensions: 6 images (3 all-zero, 3 diagonal 255).
2. Clean passage and validation for both JSON and CSV loaders.
3. Majority-class model learns solely from training records.
4. Objective held-out metric evaluation on unseen test samples.
5. Rejection of un-fitted predictions (`NotFittedError`).
6. Rejection of malformed pixel dimensions, out-of-range pixels, or invalid labels (`SchemaError`).
7. Rejection of overlapping train/test record IDs.

---

## 5. Real-Data Extension Guidance

If extending this baseline to real-world image datasets (e.g. MNIST / Fashion-MNIST):
- **Source:** TensorFlow Datasets (`tfds.load('mnist')`) or Keras (`tf.keras.datasets.mnist`).
- **Licence:** Creative Commons Attribution-Share Alike 3.0 / MIT.
- **Split Design:** 60,000 train / 10,000 test with majority class representing $\approx 10\%$ per digit class (the honest random/majority baseline accuracy is $\approx 10.0\%$).
