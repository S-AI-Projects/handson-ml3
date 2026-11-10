# R33-T054: Bounded CPU smoke run (notebook 12, Scalars)

## Data contract
Six SYNTHETIC 4x4 grayscale images (3 all-zero = label 0, 3 with white
diagonal 255 = label 1). Schema: record_id, group_id, pixels, label.
Group-disjoint splits: train=g1, val=g2, test=g3. CPU-only, offline.

## Smoke run
Pure-Python logistic baseline, fixed seed 0, max 32 samples, max 20 update
steps (full batch, lr 0.5). Per-sample loss is a scalar float (binary
cross-entropy). Tests assert interface correctness: finite loss, one
prediction per sample in [0,1], recorded step count = 20, determinism, and
rejection of >32 samples, >20 steps or empty input. An optional TensorFlow
check (tf.constant(42).shape == ()) runs only if TensorFlow is installed.

## Limits
Synthetic fixture = pipeline verification only. NOT evidence of real-world
or clinical performance. No real dataset, GPU or credentials are used.
Any real-data extension must record source, licence, access requirements
and split design separately.

## Run
python3 -m unittest discover -s course-labs/r33-t054 -p "test_*.py" -v
