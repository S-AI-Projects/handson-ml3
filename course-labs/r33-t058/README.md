# R33-T058: Error-analysis report (notebook 01, Generating the data and figures)

## Data contract
Six SYNTHETIC rows: x1 = [0,1,2,3,4,5], x2 = 1, target = 2*x1 + 1
(a deterministic formula fixture). Schema: record_id (r1..r6), x1, x2, target.
Split (explicit): train = r1..r4, validation = r5, test = r6.
These are NOT observations from the named real dataset.

## Model
Least-squares line on x1 fitted on the TRAINING rows only (x2 is constant,
so it is ignored). It recovers slope 2.0 and intercept 1.0, so every clean
prediction is exact.

## Report
One row per record: record_id, split, expected, predicted, error
(predicted - expected), abs_error, reason_category
(correct / over_prediction / under_prediction / injected_error).
Summary: n_total, n_correct, n_errors, mae, per-split and per-reason counts,
and error_record_ids.

## Deliberately wrong prediction
To make the report testable, r6 (expected 11.0) is injected with predicted
9.0. It appears in the report as record r6 (error -2.0, injected_error);
the five correct predictions do not appear in the error list. Aggregate
counts are checked against the detailed rows (6 total, 5 correct, 1 error,
MAE = 2/6).

## Limits
Synthetic fixture = pipeline verification only. NOT evidence of real-world
or clinical performance. The injected error is a test hook, not a model
failure. Any real-data extension must record source, licence, access
requirements and split design separately.

## Run
python3 -m unittest discover -s course-labs/r33-t058 -p "test_*.py" -v
