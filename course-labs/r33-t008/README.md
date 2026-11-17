# Course Lab R33-T008: Error-Analysis Reporting for Linear Regression

**Student:** NAKSHATRA MESHRAM  
**Task:** R33-T008 (8 of 10)  
**Notebook Item:** `04_training_linear_models.ipynb`  
**Target Section:** `# Learning Curves`  
**Repository:** [S-AI-Projects/handson-ml3](https://github.com/S-AI-Projects/handson-ml3)

---

## 1. Overview & Objective

In Chapter 04 (*Training Linear Models*), the **Learning Curves** section investigates model diagnostics, underfitting, overfitting, and residual errors during training.

This lab provides an automated, structured **error-analysis reporting engine** (`ErrorAnalysisReport`) that inspects regression predictions against expected ground-truth targets:
1. **Error Reporting Contract:** For any prediction exceeding an error tolerance threshold ($\tau = 0.01$), an `ErrorRecord` is generated containing stable `record_id`, `expected_value`, `predicted_value`, residual metrics, and a diagnostic `reason_category`.
2. **Exclusion of Correct Predictions:** Accurate predictions ($|y - \hat{y}| \le \tau$) are counted as correct and strictly excluded from the detailed error records.
3. **Aggregate Count Invariant:** The aggregate error summary counts match the detailed error rows exactly ($\sum \text{reason\_counts} = \text{total\_errors} = \text{len}(\text{error\_records})$).
4. **Fault Injection Testing:** Validates that an injected faulty prediction on record `r6` is explicitly flagged, accurately isolated, and properly attributed.

> **Notice:** The 6-row synthetic fixture is designed exclusively for **pipeline verification** and error report testability. It is not empirical evidence of real-world or clinical performance.

---

## 2. Error-Analysis Report Schema & Metrics

### Record-Level Attributes (`ErrorRecord`):
- `record_id`: Stable identifier (e.g. `r6`).
- `split`: Dataset partition (`train`, `validation`, `test`).
- `x1`, `x2`: Feature values.
- `expected_value`: Ground truth target value ($2 \cdot x_1 + 1$).
- `predicted_value`: Model prediction or injected anomaly.
- `residual`: Signed error ($\hat{y} - y$).
- `absolute_error`: $|\hat{y} - y|$.
- `relative_error`: $|\hat{y} - y| / |y|$.
- `reason_category`: Diagnostic classification (`sensor_saturation_artifact`, `injected_fault`, `extrapolation_uncertainty`).

### Report-Level Invariants:
- $\text{total\_evaluated} = \text{total\_correct} + \text{total\_errors}$
- $\text{total\_errors} = \text{len}(\text{error\_records})$
- $\sum_{k} \text{aggregate\_counts\_by\_reason}[k] = \text{total\_errors}$

---

## 3. Synthetic Fixture Specification

Six synthetic rows obeying $\text{target} = 2 \cdot x_1 + 1$ with $x_2 = 1.0$:
- **`r1`**: $x_1 = 0, x_2 = 1, \text{target} = 1$ (Train)
- **`r2`**: $x_1 = 1, x_2 = 1, \text{target} = 3$ (Train)
- **`r3`**: $x_1 = 2, x_2 = 1, \text{target} = 5$ (Train)
- **`r4`**: $x_1 = 3, x_2 = 1, \text{target} = 7$ (Train)
- **`r5`**: $x_1 = 4, x_2 = 1, \text{target} = 9$ (Validation)
- **`r6`**: $x_1 = 5, x_2 = 1, \text{target} = 11$ (Test)

When trained on `r1..r4`, the closed-form normal equations solve:
$$w_1 = 2.0, \quad w_2 = 1.0, \quad \text{MSE} = 0.0$$

---

## 4. Verification Commands

Run unit test discovery:
```bash
python3 -m unittest discover -s course-labs/r33-t008 -p "test_*.py" -v
```

Run comprehensive verification runner:
```bash
python3 work/R33-T008/run_verification.py
```
