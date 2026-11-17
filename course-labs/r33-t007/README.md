# Course Lab R33-T007: Reproducible Configuration and Parameter Schema

**Student:** NAKSHATRA MESHRAM  
**Task:** R33-T007 (7 of 10)  
**Notebook Item:** `02_end_to_end_machine_learning_project.ipynb`  
**Target Section:** `## Looking for Correlations`  
**Repository:** [S-AI-Projects/handson-ml3](https://github.com/S-AI-Projects/handson-ml3)

---

## 1. Overview & Objective

In Chapter 02 (*End-to-End Machine Learning Project*), the **Looking for Correlations** section inspects linear feature-target correlations across dataset attributes.

This lab provides a strict, serializable configuration system (`ExperimentConfig`) ensuring 100% reproducible data partitioning, model initialization, feature ordering, and predictions:
1. **Serializable Configuration:** Encapsulates random seed, schema version (`1.0.0`), explicit feature ordering (`["x1", "x2"]`), target attribute, deterministic split IDs, and algorithm hyperparameters.
2. **Strict Validation & Rejection:** Rejects unknown fields, contradictory split IDs (e.g. partition overlaps), incompatible feature shapes, and invalid hyperparameters.
3. **Exact Reproducibility:** Serializing to JSON and reloading via `ExperimentConfig.from_json()` deterministically reconstructs identical dataset splits, parameter weights, and inference outputs.

> **Notice:** The 6-row synthetic regression fixture is designed exclusively for **pipeline verification** and reproducibility testing. It is not an empirical substitute for the California Housing dataset.

---

## 2. Configuration Schema & Contracts

### Top-Level Configuration Fields:
- `schema_version`: String identifying configuration schema version (must be `"1.0.0"`).
- `seed`: Integer pseudo-random seed for deterministic execution.
- `feature_ordering`: Ordered list of input feature names (`["x1", "x2"]`).
- `target_column`: Name of target attribute (`"target"`).
- `split_ids`: Dictionary mapping partition names (`train`, `validation`, `test`) to explicit lists of record IDs.
- `algorithm_settings`: Hyperparameter dictionary (`algorithm`, `fit_intercept`, `regularization_alpha`, `max_iter`, `tolerance`).

### Validation & Rejection Matrix:

| Condition | Example Violation | Failure Raised |
|---|---|---|
| Unknown top-level key | `{"extra_arg": 123}` | `ConfigValidationError` |
| Unknown algorithm setting | `{"optimizer": "adam"}` | `ConfigValidationError` |
| Contradictory split IDs | `r1` in both `train` and `validation` | `ConfigValidationError` |
| Unknown split ID | `r99` not in dataset | `ConfigValidationError` |
| Incompatible feature shape | 3 features supplied to 2-feature regressor | `SchemaError` |
| Negative regularization | `regularization_alpha = -1.0` | `ConfigValidationError` |
| Non-positive iterations | `max_iter = 0` | `ConfigValidationError` |

---

## 3. Synthetic Fixture Specification

Six synthetic rows obeying the exact linear formula $\text{target} = 2 \cdot x_1 + 1$:
- **`r1`**: $x_1 = 0, x_2 = 1, \text{target} = 1$ (Train)
- **`r2`**: $x_1 = 1, x_2 = 1, \text{target} = 3$ (Train)
- **`r3`**: $x_1 = 2, x_2 = 1, \text{target} = 5$ (Train)
- **`r4`**: $x_1 = 3, x_2 = 1, \text{target} = 7$ (Train)
- **`r5`**: $x_1 = 4, x_2 = 1, \text{target} = 9$ (Validation)
- **`r6`**: $x_1 = 5, x_2 = 1, \text{target} = 11$ (Test)

Closed-form normal equations on training set $r1..r4$ yield exact weights:
$$w_{x_1} = 2.0, \quad w_{x_2} = 1.0, \quad \text{MSE} = 0.0$$

---

## 4. Verification Commands

Run unit test discovery:
```bash
python3 -m unittest discover -s course-labs/r33-t007 -p "test_*.py" -v
```

Run comprehensive verification runner:
```bash
python3 work/R33-T007/run_verification.py
```
