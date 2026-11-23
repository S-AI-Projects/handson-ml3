# Course Lab R33-T026: Bounded CPU Smoke Run

## Overview
Implements a bounded CPU-only training and inference smoke run on an 8-row classification fixture, with finite outputs, step counts <= 20, and zero GPU/external dependencies.

## Smoke Run Contract
- Max samples: 8 rows (strictly <= 32).
- Update steps: 15 (strictly <= 20).
- Model: CPU standard-library deterministic logistic model.
- Metrics asserted: Finite binary cross-entropy loss and prediction shape alignment.

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t026 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
