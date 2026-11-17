# Course Lab R33-T010: Saved-Pipeline Interface and Schema Validation

**Student:** NAKSHATRA MESHRAM  
**Task:** R33-T010 (10 of 10)  
**Notebook Item:** `02_end_to_end_machine_learning_project.ipynb`  
**Target Section:** `## Data Cleaning`  
**Repository:** [S-AI-Projects/handson-ml3](https://github.com/S-AI-Projects/handson-ml3)

---

## 1. Overview & Objective

In Chapter 02 (*End-to-End Machine Learning Project*), the **Data Cleaning** section covers data preprocessing, pipelines, and preparing models for persistent deployment.

This lab delivers a safe, auditable **saved-pipeline interface** (`SavedLinearPipeline`):
1. **Safe Serialization (Pickle-Free):** Serializes model configurations, feature schema metadata, and fitted regression weights to a clean JSON artifact. Untrusted pickle formats are strictly avoided to eliminate arbitrary code execution vulnerabilities.
2. **Strict Prediction Schema Validation:** Exposes a prediction function that enforces strict feature contracts:
   - Missing expected features raise `SchemaValidationError`.
   - Unknown extra attributes raise `SchemaValidationError`.
   - Non-numeric or non-finite values (`NaN`, `Inf`) raise `SchemaValidationError`.
3. **Parity Guarantee:** Predictions before saving and after reloading from file agree within an explicit tolerance ($\epsilon = 10^{-9}$).
4. **Isolated Inference:** Reloaded pipeline requires zero training data, zero internet access, and zero external credentials at prediction time.
5. **Clear Version Rejection:** Unsupported artifact schema versions raise `VersionMismatchError`, and corrupted artifacts raise `ModelCorruptionError`.

> **Notice:** The 6-row synthetic regression fixture is designed exclusively for **pipeline verification** and saved-model interface testing. It is not an empirical substitute for real-world benchmark datasets.

---

## 2. Artifact Schema & Architecture

### Saved JSON Artifact (`model.json`):
```json
{
  "format": "safe_json_pipeline",
  "schema_version": "1.0.0",
  "feature_schema": {
    "feature_names": ["x1", "x2"],
    "feature_types": {"x1": "float", "x2": "float"}
  },
  "parameters": {
    "weights": [2.0, 1.0],
    "intercept": 0.0
  },
  "metadata": {
    "num_training_samples": 4,
    "training_split_ids": ["r1", "r2", "r3", "r4"],
    "formula": "target = 2 * x1 + 1",
    "is_synthetic": true
  }
}
```

### Schema Validation & Failure Modes:

| Test Condition | Failure Type | Behavior |
|---|---|---|
| Missing feature (e.g. `{"x1": 5.0}`) | `SchemaValidationError` | Rejects missing `x2` |
| Unknown feature (e.g. `{"x1": 5.0, "x2": 1.0, "extra": 42}`) | `SchemaValidationError` | Rejects extra key |
| Non-numeric type (e.g. string) | `SchemaValidationError` | Rejects string |
| Non-finite value (`NaN` / `Inf`) | `SchemaValidationError` | Rejects non-finite |
| Unsupported version (`"99.0.0"`) | `VersionMismatchError` | Rejects incompatible version |
| Malformed JSON syntax | `ModelCorruptionError` | Rejects corrupted file |

---

## 3. Synthetic Fixture Specification

Six synthetic rows obeying $\text{target} = 2 \cdot x_1 + 1$ with $x_2 = 1.0$:
- **`r1`**: $x_1 = 0, x_2 = 1, \text{target} = 1$ (Train)
- **`r2`**: $x_1 = 1, x_2 = 1, \text{target} = 3$ (Train)
- **`r3`**: $x_1 = 2, x_2 = 1, \text{target} = 5$ (Train)
- **`r4`**: $x_1 = 3, x_2 = 1, \text{target} = 7$ (Train)
- **`r5`**: $x_1 = 4, x_2 = 1, \text{target} = 9$ (Validation)
- **`r6`**: $x_1 = 5, x_2 = 1, \text{target} = 11$ (Test)

Closed-form normal equations on training set `r1..r4` learn exact weights:
$$w_{x_1} = 2.0, \quad w_{x_2} = 1.0, \quad \text{MSE} = 0.0$$

---

## 4. Verification Commands

Run unit test discovery:
```bash
python3 -m unittest discover -s course-labs/r33-t010 -p "test_*.py" -v
```

Run comprehensive verification runner:
```bash
python3 work/R33-T010/run_verification.py
```
