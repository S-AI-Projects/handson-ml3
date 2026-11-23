# Course Lab R33-T028: Perturbation Robustness

## Overview
Evaluates robustness of regularized linear regression interface under input perturbations (record order permutations, missing fields, and shape mismatches).

## Fixture Specification
- Six synthetic rows: `x1=[0, 1, 2, 3, 4, 5]`, `x2=1.0`, formula `target = 2*x1 + 1`.
- Partitioned into train `r1..r4`, validation `r5`, and test `r6`.
- Verified record-order permutation invariance via stable record ID mapping.

## Verification Checks
Run offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r33-t028 -p "test_*.py" -v
```

Synthetic fixture results are for pipeline verification only, not evidence of real-world performance.
