# R33-T014: Verify the Evaluation Metric

## Purpose

This small, standard-library-only component independently verifies
classification precision, recall, and regression mean absolute error (MAE).

## Data contract

The fixture contains eight synthetic records with:

- `record_id`: unique row identifier
- `group_id`: unique group identifier
- `x1`: values 0 through 7
- `x2`: constant value 1
- `label`: four zeros followed by four ones

These are simulated values for pipeline verification. They are not
observations from any named real dataset and do not support real-world,
clinical, or predictive-performance claims.

## Fixed stratified split

- Train: r1, r2, r5, r6
- Validation: r3, r7
- Test: r4, r8

Each split contains equal numbers of class 0 and class 1. Every record and
group is assigned to only one split.

## Classification metric definition

Precision is TP / (TP + FP). Recall is TP / (TP + FN).
The implementation reports positive-class (label 1) binary metrics,
not macro or weighted averages.

If either denominator is zero, the corresponding metric is defined as 0.0.

For truth [1, 1, 0, 0] and prediction [1, 0, 1, 0]:

- TP = 1
- FP = 1
- FN = 1
- Precision = 0.5
- Recall = 0.5

## Regression metric definition

MAE is the arithmetic mean of absolute prediction errors.

For truth [1, 2, 3] and prediction [1, 3, 2], MAE = 2/3.

## Reproducibility and limits

The fixture is deterministic and uses seed 42 as metadata. The implementation
uses Python's standard library only, runs offline on CPU, and requires no
external model or dataset.

This is metric/pipeline verification, not evidence of real-world accuracy.
