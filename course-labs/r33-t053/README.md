# R33-T053: Evaluation metric check (Leaky ReLU, notebook 11)

## Data contract
Six SYNTHETIC 4x4 grayscale images (3 all-zero = label 0, 3 with white
diagonal 255 = label 1). Schema: record_id, group_id, pixels, label.
Splits are group-disjoint: train=g1, val=g2, test=g3. CPU-only, offline.

## Metric definitions
- Precision = TP/(TP+FP); Recall = TP/(TP+FN); MAE = mean(|y - yhat|).
- Zero denominator -> 0.0 (configurable via `zero_division`).
- Averaging: binary (positive=1), macro (mean over classes), micro (pooled).
- Worked examples: truth[1,1,0,0]/pred[1,0,1,0] -> P=R=0.5;
  truth[1,2,3]/pred[1,3,2] -> MAE=2/3.
- `leaky_relu` default alpha=0.2 matches the notebook's LeakyReLU(alpha=0.2).

## Limits
Synthetic fixture = pipeline verification only. It is NOT evidence of
real-world or clinical performance. No real dataset is used.
Any optional real-data extension must record source, licence, access
requirements and split design separately.

## Run
python3 -m unittest discover -s course-labs/r33-t053 -p "test_*.py" -v
