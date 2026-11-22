# R33-T040: Define a Saved-Pipeline Interface

**Task:** R33-T040 · 10 of 10  
**Student:** Snehakumari Vinod Singh  
**Section:** 14 deep computer vision with cnns / Classification and Localization  
**Notebook:** `14_deep_computer_vision_with_cnns.ipynb`  
**Branch:** `student/e26b07f0590/R33-T040`

---

## Goal

Save and reload the small image pipeline configuration/parameters and expose a prediction function with strict schema validation:
- Safe JSON serialization (never load untrusted pickle).
- Rejection of schema/version mismatches and unauthorized keys.
- Strict input validation (record structure, 4x4 image shape, pixel value range `[0..255]`).
- Self-contained inference: no training data, GPU, or external credentials required at inference time.
- Prediction agreement before and after reload within explicit numerical tolerance ($10^{-6}$).

---

## Synthetic Fixture

> **⚠️ Disclaimer:** These records are **fully synthetic** — generated for saved-pipeline interface verification only. Results here are **not** evidence of real-world, clinical, or production performance and must not be cited as such.

Six synthetic 4×4 grayscale images (16 flattened pixels):
- Three all-zero images labelled `0` (`group_id = "g0"`).
- Three white-diagonal images labelled `1` (`group_id = "g1"`).

| record_id | group_id | label | Description |
|-----------|----------|-------|-------------|
| img1      | g0       | 0     | All zeros (4x4) |
| img2      | g0       | 0     | All zeros (4x4) |
| img3      | g0       | 0     | All zeros (4x4) |
| img4      | g1       | 1     | White diagonal (indices 0, 5, 10, 15 are 255) |
| img5      | g1       | 1     | White diagonal (indices 0, 5, 10, 15 are 255) |
| img6      | g1       | 1     | White diagonal (indices 0, 5, 10, 15 are 255) |

---

## Saved Interface & Schema Contract

### Serialized State Format (`pipeline_state.json`)
```json
{
  "schema_version": "1.0.0",
  "model_type": "linear_image_diagonal_filter",
  "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  "bias": -2.0,
  "threshold": 0.5,
  "input_shape": [4, 4],
  "metadata": {
    "description": "Safe JSON serialized image classifier",
    "training_data_required_at_inference": false,
    "credentials_required": false
  }
}
```

### Strict Rejections
1. **Unsafe Deserialization:** Only standardized JSON format is permitted; Python pickle is strictly excluded.
2. **Schema Versioning:** Rejects mismatch with `SUPPORTED_SCHEMA_VERSION` (`1.0.0`).
3. **Pixel Dimensionality:** Enforces exactly 16 values (4×4 matrix).
4. **Range Bounds:** Rejects negative values or values exceeding 255.0.

---

## Running Checks

Run unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t040 -p "test_*.py" -v
```

Run verification runner:
```bash
python3 work/R33-T040/run_verification.py
```

---

## Limits & Data Contract

- Standard library only — no OpenCV, PIL, TensorFlow, or PyTorch required for interface verification.
- Validates model serialization and input schema validation contracts.

## Optional Real-Data Extension

Any optional real-world computer vision extension must record:
- Source image dataset name, resolution, and licensing
- Normalization (mean/std subtraction) and bounding-box coordinates format
- Exact inference runtime latency and memory constraints
