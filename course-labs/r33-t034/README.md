# Course Lab R33-T034: Evaluation Metric Verification & Text Fixture Cross-Checks

**Student:** Snehakumari Vinod Singh  
**Task Code:** R33-T034 (Task 4 of 10)  
**Assigned Branch:** `student/e26b07f0590/R33-T034`  
**Related Issue:** [Issue #34](https://github.com/S-AI-Projects/handson-ml3/issues/34)  
**Target Item:** `16_nlp_with_rnns_and_attention.ipynb` (Generating Fake Shakespearean Text)

---

## 1. Scope & Objective

This contribution provides a verified, CPU-only evaluation metric validation module and synthetic text classification fixture for natural language processing contracts in `16_nlp_with_rnns_and_attention.ipynb`. It mathematically formalizes precision, recall, F1, accuracy, and mean absolute error (MAE), explicitly specifying averaging behavior and zero-denominator safety handling.

> ### ⚠️ Synthetic Text Fixture Disclaimer
> The six fictional text records in `fixture.json` and `fixture.csv` (`t1` through `t6`) are designed strictly for pipeline schema and metric correctness verification.
>
> **These are NOT observations from any real-world NLP corpus or Shakespearean literary text.** They are mock verification fixtures and must never be cited as evidence of real-world or clinical predictive performance.

---

## 2. Evaluation Metric Formulations & Explicit Edge Cases

### Classification Metrics
- **Precision:**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  *Zero-denominator rule:* If $\text{TP} + \text{FP} = 0$ (no positive predictions), precision returns `zero_division` (default $0.0$).
- **Recall:**
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Zero-denominator rule:* If $\text{TP} + \text{FN} = 0$ (no positive ground truths), recall returns `zero_division` (default $0.0$).
- **F1 Score:**
  $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
  *Zero-denominator rule:* Returns $0.0$ if $\text{Precision} + \text{Recall} = 0$.

### Regression Metric
- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

---

## 3. Explicit Benchmark Verification Cases

1. **Classification Case:**
   - Ground Truth: `[1, 1, 0, 0]`
   - Prediction: `[1, 0, 1, 0]`
   - Confusion Matrix: $\text{TP}=1, \text{FP}=1, \text{FN}=1, \text{TN}=1$
   - $\text{Precision} = 1 / (1 + 1) = \mathbf{0.5}$
   - $\text{Recall} = 1 / (1 + 1) = \mathbf{0.5}$
   - $\text{F1} = \mathbf{0.5}$, $\text{Accuracy} = \mathbf{0.5}$

2. **Regression Case:**
   - Ground Truth: `[1.0, 2.0, 3.0]`
   - Prediction: `[1.0, 3.0, 2.0]`
   - $\text{MAE} = \frac{|1-1| + |2-3| + |3-2|}{3} = \frac{0 + 1 + 1}{3} = \mathbf{\frac{2}{3}} \approx 0.666667$

---

## 4. Synthetic Text Fixture Specification

| Record ID | Text | Label | Role |
| :--- | :--- | :--- | :--- |
| `t1` | `red apple` | `1` | Positive sample |
| `t2` | `green apple` | `1` | Positive sample |
| `t3` | `blue sky` | `0` | Negative sample |
| `t4` | `dark sky` | `0` | Negative sample |
| `t5` | `red fruit` | `1` | Positive sample |
| `t6` | `blue cloud` | `0` | Negative sample |

---

## 5. Running Automated Checks & Tests

### Run Unit Tests
```bash
python3 -m unittest discover -s course-labs/r33-t034 -p "test_*.py" -v
```

### Run Verification Runner (Tool versions, evidence, exit code 0)
```bash
python3 work/R33-T034/run_verification.py
```
Outputs:
- Generates `work/R33-T034/tool_versions.json`
- Generates `work/R33-T034/unittest_output.txt`
- Generates `work/R33-T034/evidence.json`
- Exits with code `0` on success or non-zero on assertion failure.
