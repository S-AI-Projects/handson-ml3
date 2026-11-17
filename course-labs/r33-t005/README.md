# Course Lab R33-T005: Preprocessing Contract & Parameter Isolation

**Student:** NAKSHATRA MESHRAM  
**Task Code:** R33-T005 (Task 5 of 10)  
**Assigned Branch:** `student/e26b07f0682` / `R33-T005`  
**Target Item:** `04_training_linear_models.ipynb` (Section: Stochastic Gradient Descent)

---

## 1. Scope & Objective

This contribution provides an explicit preprocessing contract for feature scaling and missing-value imputation under `04_training_linear_models.ipynb` (Stochastic Gradient Descent). It guarantees:
1. **Parameter Isolation:** Scaling means ($\mu_{\text{train}}$), standard deviations ($\sigma_{\text{train}}$), and imputation values are computed strictly from training rows (`r1..r4`).
2. **Transform Immutability:** Calling `transform()` on held-out inputs (`r5`, `r6`) cannot and does not alter fitted statistics.
3. **Contamination Rejection:** Attempting to fit on validation/test splits or contaminated records raises an immediate `ContaminationError`.
4. **Deterministic Feature Schema:** Retains exact float data types and strict feature column ordering `["x1", "x2"]`.

> ### ⚠️ Synthetic Formula Fixture Disclaimer
> The 6-row regression dataset in `fixture.json` and `fixture.csv` is generated deterministically from the linear formula:
> $$\text{target} = 2 \cdot x_1 + 1, \quad x_2 = 1, \quad x_1 \in \{0, 1, 2, 3, 4, 5\}$$
> 
> **These are NOT observations from any real-world dataset.** They serve solely as a deterministic fixture to verify preprocessing contracts, shape/dtype integrity, and leakage prevention.

---

## 2. Preprocessing Specification

### 2.1 Training Statistics (Computed on `r1..r4`, $x_1 \in [0, 1, 2, 3]$)
- **Feature $x_1$:**
  - Mean: $\mu_{\text{train}} = \frac{0 + 1 + 2 + 3}{4} = 1.5$
  - Variance: $\sigma^2_{\text{train}} = \frac{(-1.5)^2 + (-0.5)^2 + (0.5)^2 + (1.5)^2}{4} = 1.25$
  - Standard Deviation: $\sigma_{\text{train}} = \sqrt{1.25} \approx 1.118034$
- **Feature $x_2$:**
  - Mean: $1.0$, Standard Deviation: $1.0$ (constant feature zero-variance fallback divisor).

### 2.2 Transformed Values
| Split | Record ID | Raw $x_1$ | Standardized $x_1 = (x_1 - 1.5)/\sqrt{1.25}$ | Raw $x_2$ | Standardized $x_2$ | Target |
|---|---|---|---|---|---|---|
| Train | `r1` | 0 | $\approx -1.341641$ | 1 | 0.0 | 1 |
| Train | `r2` | 1 | $\approx -0.447214$ | 1 | 0.0 | 3 |
| Train | `r3` | 2 | $\approx +0.447214$ | 1 | 0.0 | 5 |
| Train | `r4` | 3 | $\approx +1.341641$ | 1 | 0.0 | 7 |
| Validation | `r5` | 4 | $\approx +2.236068$ | 1 | 0.0 | 9 |
| Test | `r6` | 5 | $\approx +3.130495$ | 1 | 0.0 | 11 |

- **Shape:** 2-dimensional feature vector `[x1, x2]` per record.
- **Type:** IEEE 754 `float`.

---

## 3. Contamination & Safeguard Rules

1. **Held-out Split Rejection:** `fit()` checks record split flags and IDs. If any record marked `validation`, `test`, `r5`, or `r6` is passed to `fit()`, a `ContaminationError` is raised.
2. **Deliberate Contamination Flags:** Records carrying `_contaminated=True` or infinite values (`inf`, `-inf`) are rejected.
3. **Read-Only Transform:** `transform()` has no mutation access to `self.state_`.

---

## 4. Automated Verification & Testing

Execute the test suite using standard library `unittest` (offline, CPU-only):

```bash
python3 -m unittest discover -s course-labs/r33-t005 -p "test_*.py" -v
```

The test suite asserts:
1. Exact row count ($6$), splits, and formula values.
2. Training-only parameter fitting ($\mu=1.5, \sigma=\sqrt{1.25}$).
3. Transformed output shape (2 features), dtype (`float`), and standardized numerical values.
4. Parameter immutability when transforming held-out sets.
5. Rejection of held-out records supplied to `fit()`.
6. Rejection of deliberate contamination fixtures.
7. Mean imputation of missing values using training statistics.

---

## 5. Real-Data Extension Guidance

If integrating real SGD regression data:
- **Source:** Scikit-Learn California Housing dataset.
- **Pipeline:** Embed `ExplicitPreprocessor` within `sklearn.pipeline.Pipeline` or a custom TensorFlow preprocessing layer to freeze normalization weights prior to gradient descent.
