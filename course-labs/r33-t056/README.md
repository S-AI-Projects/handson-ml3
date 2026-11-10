# R33-T056: Saved-pipeline interface (notebook 17, Variational Autoencoder)

## Data contract
Six SYNTHETIC 4x4 grayscale images (3 all-zero = label 0, 3 with white
diagonal 255 = label 1). Schema: record_id, group_id, pixels, label.
Group-disjoint splits: train=g1, val=g2, test=g3. CPU-only, offline.
NOT the named real dataset.

## Saved format (JSON only)
{"schema_version": 1, "model_type": "nearest_centroid",
 "config": {"image_size": 4, "pixel_max": 255},
 "params": {"centroids": {"0": [16 floats], "1": [16 floats]}}}
Pickle is never used or loaded: a pickle file is rejected as invalid JSON.

## Interface
- fit(train_records) -> model dict (per-class mean image, scaled 0..1)
- save(model, path) / load(path): validate schema on both sides
- predict(model, pixels) -> {"label", "scores"}; validates shape and range
- Any schema_version / model_type / config / key / length / non-finite
  problem raises PipelineSchemaError with a clear message.

## Result
Before-save and after-reload predictions agree within TOL = 1e-9 on all six
records (labels identical). Inference needs only the saved JSON: no training
data, credentials or network.

## Limits
This is a deliberately tiny stand-in for a notebook-17 autoencoder
pipeline; no autoencoder is trained. Synthetic fixture = pipeline
verification only, NOT evidence of real-world or clinical performance.
Any real-data extension must record source, licence, access requirements
and split design separately.

## Run
python3 -m unittest discover -s course-labs/r33-t056 -p "test_*.py" -v
