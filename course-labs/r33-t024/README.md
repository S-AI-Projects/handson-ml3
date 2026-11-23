# Course Lab R33-T024: Metric Verification

## Overview
Cross-checks evaluation metrics for synthetic image classification and regression fixtures with explicit zero-denominator handling.

## Verified Metric Specifications
- **Classification Contract**: truth `[1, 1, 0, 0]`, prediction `[1, 0, 1, 0]` produces `precision = 0.5`, `recall = 0.5`.
- **Regression Contract**: truth `[1, 2, 3]`, prediction `[1, 3, 2]` produces `MAE = 2/3`.
- **Zero Denominator Handling**: Gracefully defaults to defined `zero_division` constant (0.0).

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t024 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
