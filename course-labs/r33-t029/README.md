# Course Lab R33-T029: Saved Pipeline Interface

## Overview
Implements a safe JSON-based model serialization interface (strictly avoiding unsafe pickling) paired with input schema validation for tabular classification fixtures.

## Pipeline Interface Contract
- **Storage**: Safe JSON serialization (`weights`, `bias`, `threshold`, format tag).
- **Security**: Rejects untrusted formats (e.g. `.pkl` / pickle) and unverified format tags.
- **Input Schema**: Validates mandatory `record_id`, `group_id`, `x1`, and `x2` numeric fields.

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t029 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
