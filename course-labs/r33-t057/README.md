# R33-T057: Reproducible configuration (notebook 13, Using the Dataset with Keras)

## Data contract
Six SYNTHETIC 4x4 grayscale images (3 all-zero = label 0, 3 with white
diagonal 255 = label 1). Schema: record_id, group_id, pixels, label.
Declared groups: g1, g2, g3 (one image of each class per group).
NOT the named real dataset.

## Configuration (plain JSON)
schema_version, seed, input_shape [4,4], feature_order (p00..p33,
row-major), split_ids (record IDs for train/val/test), algorithm
{name: nearest_centroid, pixel_max: 255}.
The seed deterministically assigns whole groups to train/val/test, so the
split is group-disjoint and stratified; the resulting IDs are stored in the
config, so reloading reproduces the exact same split without the seed.

## Validation (ConfigError)
Rejected: unknown or missing fields, wrong schema_version, bad seed,
incompatible input_shape, duplicate/short feature_order, unknown algorithm
or pixel_max, a record in two splits, unknown record IDs, group leakage
across splits, and a train split without both classes.
Inputs of the wrong shape or pixel range are rejected at feature extraction.

## Result
A config saved to JSON and reloaded reconstructs the same split and gives
identical predictions (tolerance 1e-9). Environment versions are recorded
by environment_info() (saved to work/R33-T057/environment.json).

## Limits
Synthetic fixture = pipeline verification only. NOT evidence of real-world
or clinical performance. No Keras dataset is loaded; this is a
bounded stand-in for the configuration part of the notebook section.
Any real-data extension must record source, licence, access requirements
and split design separately.

## Run
python3 -m unittest discover -s course-labs/r33-t057 -p "test_*.py" -v
