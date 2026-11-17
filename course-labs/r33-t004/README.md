# Course Lab R33-T004: Metric Correctness, Explicit Averaging, and Zero-Denominator Handling

**Student:** NAKSHATRA MESHRAM  
**Task Code:** R33-T004 (Task 4 of 10)  
**Assigned Branch:** `student/e26b07f0682` / `R33-T004`  
**Target Item:** `06_decision_trees.ipynb` (Section: Regularization Hyperparameters)

---

## 1. Scope & Objective

This contribution provides an independent, mathematically rigorous metric verification module for evaluating regularized classifiers under `06_decision_trees.ipynb`. It formally defines:
1. Exact confusion matrix computation ($TP, FP, TN, FN$).
2. Precision, Recall, $F_1$, and Accuracy formulation with explicitly specified zero-denominator edge cases.
3. Canonical regression MAE / MSE benchmark validation.
4. Decision tree stump regularization evaluation over an 8-row synthetic fixture.

> ### ⚠️ Synthetic Fixture Disclaimer
> The 8-row dataset in `fixture.json` and `fixture.csv` contains purely synthetic numerical values:
> - $x_1 \in [0, 7]$, $x_2 = 1$, and $\text{label} \in \{0, 0, 0, 0, 1, 1, 1, 1\}$.
> 
> **These are NOT observations from any real-world dataset.** They serve solely as a bounded verification fixture for metric calculations and pipeline CI checks.

---

## 2. Metric Specifications & Canonical Benchmarks

### 2.1 Canonical Classification Case
Given:
$$\mathbf{y}_{\text{true}} = [1, 1, 0, 0], \quad \mathbf{y}_{\text{pred}} = [1, 0, 1, 0]$$

Confusion Matrix:
- $TP = 1$, $FP = 1$, $TN = 1$, $FN = 1$

Metrics:
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{1}{1 + 1} = 0.5$$
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{1}{1 + 1} = 0.5$$
$$F_1 = 2 \cdot \frac{0.5 \cdot 0.5}{0.5 + 0.5} = 0.5, \quad \text{Accuracy} = \frac{1 + 1}{4} = 0.5$$

### 2.2 Canonical Regression Case
Given:
$$\mathbf{y}_{\text{true}} = [1, 2, 3], \quad \mathbf{y}_{\text{pred}} = [1, 3, 2]$$

Metrics:
$$\text{MAE} = \frac{|1-1| + |2-3| + |3-2|}{3} = \frac{0 + 1 + 1}{3} = \frac{2}{3} \approx 0.6667$$
$$\text{MSE} = \frac{(0)^2 + (-1)^2 + (1)^2}{3} = \frac{2}{3} \approx 0.6667$$

---

## 3. Zero-Denominator Handling Specification

When computing ratios where denominators can be zero, the implementation explicitly prevents uncaught exceptions:
- **No Positive Predictions Made ($TP + FP = 0$):**
  - $\text{Precision} = \text{zero\_division}$ (default `0.0`).
  - Flagged in result: `zero_division_applied = True`.
- **No Ground Truth Positive Samples ($TP + FN = 0$):**
  - $\text{Recall} = \text{zero\_division}$ (default `0.0`).
  - Flagged in result: `zero_division_applied = True`.
- **Zero Precision and Recall ($P + R = 0$):**
  - $F_1 = \text{zero\_division}$ (default `0.0`).

---

## 4. Decision Tree Stump Regularization Benchmark

Using the 8-row fixture ($x_1 \in [0, 7]$, threshold split at $x_1 \ge \theta$):
- **Optimal Regularization ($\theta = 3.5$):**
  - Perfectly separates negative class ($x_1 \le 3$) and positive class ($x_1 \ge 4$).
  - $\text{Accuracy} = 1.0, \text{Precision} = 1.0, \text{Recall} = 1.0, F_1 = 1.0$.
- **Over-Regularized Suboptimal Stump ($\theta = 5.5$):**
  - Conservative threshold misses $x_1 \in \{4, 5\}$, yielding $TP=2, FN=2, FP=0, TN=4$.
  - $\text{Precision} = 1.0$, $\text{Recall} = 0.5$, $\text{Accuracy} = 0.75$.

---

## 5. Automated Verification & Testing

Execute the test suite with standard library `unittest` (offline, CPU-only):

```bash
python3 -m unittest discover -s course-labs/r33-t004 -p "test_*.py" -v
```

The test suite asserts:
1. Exact canonical classification precision = recall = 0.5.
2. Exact canonical regression MAE = 2/3.
3. Explicit zero-division handling when $TP+FP=0$ or $TP+FN=0$.
4. Exact fixture dimensions: 8 synthetic rows with balanced stratified split.
5. Decision tree stump regularization comparisons.

---

## 6. Real-Data Extension Guidance

If extending this metric validator to real classification datasets (such as `make_moons` or Iris in Chapter 6):
- **Source:** Scikit-Learn `datasets.make_moons` or OpenML.
- **Licence:** BSD-3-Clause / CC0.
- **Split Design:** Stratified K-Fold cross-validation to maintain macro/micro averaging consistency.
