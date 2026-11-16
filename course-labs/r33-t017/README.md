# R33-T017: Reproducible Model Configuration

## Objective
A reproducible, CPU-only synthetic image-classification smoke pipeline.

## Dataset
Six synthetic 4x4 grayscale images: three all-zero images labelled 0 and three white-diagonal images labelled 1. Every record has a unique record_id and group_id.

## Data contract
Each record contains record_id, group_id, pixels and label. Pixel values are 0 or 255, and labels are 0 or 1.

## Configuration
config.json specifies the schema version, fixed seed, feature order, explicit train/validation/test splits and algorithm settings. Unknown or contradictory settings are rejected.

## Pipeline
pipeline.py implements a deterministic nearest-template classifier using Hamming distance. It validates the fixture and configuration before prediction.

## Tests
Run from the repository root:
python3 -m unittest discover -s course-labs/r33-t017 -p "test_*.py"

Tests cover expected predictions, reproducibility, invalid configurations, malformed data, duplicate IDs, split coverage and group leakage.

## Limitations
This is a small synthetic verification fixture, not a real-world dataset or a full reproduction of the TensorFlow chapter models. It makes no real-world or clinical performance claims.
