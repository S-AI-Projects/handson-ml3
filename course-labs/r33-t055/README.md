# R33-T055: Minimal baseline (notebook 05, Nonlinear SVM Classification)

## Data contract
Eight SYNTHETIC rows: x1 = [0..7], x2 = 1, label = [0,0,0,0,1,1,1,1].
Schema: record_id (r0..r7), group_id (g1..g4), x1, x2, label.
These are NOT observations from the named real dataset.

## Split (fixed, group-disjoint, stratified)
train = g1, g2 (r0, r1, r4, r5); val = g3 (r2, r6); test = g4 (r3, r7).
Every split has one row of each class per group.

## Baseline
Majority-class classifier. The class is learned from training labels only.
Tie rule: smallest label wins (train is 2 vs 2, so the baseline predicts 0).
Metric: accuracy = correct / total, computed on the held-out rows.
Result: val accuracy = 0.5, test accuracy = 0.5. This is the honest floor
on a balanced fixture; any model must beat it to claim signal.

## Limits
Synthetic fixture = pipeline verification only. NOT evidence of real-world
or clinical performance. No real-dataset accuracy claim is made. No SVM is
trained here. Any real-data extension must record source, licence, access
requirements and split design separately.

## Run
python3 -m unittest discover -s course-labs/r33-t055 -p "test_*.py" -v
