# KuaiRec workflow helpers

This directory collects helper scripts and documentation for running the
baseline model on the [KuaiRec](https://kuairec.com/) dataset.  Place the
extracted CSV files inside `kuairec/data/` (or point the helpers to your own
location) and use the wrapper scripts in `kuairec/train/` and `kuairec/test/`
to launch training and inference with sensible defaults.

## Expected data layout

```
kuairec/
  data/
    small_matrix.csv          # or big_matrix.csv
    user_features.csv         # optional, improves feature coverage
    item_categories.csv       # optional, provides item side information
```

Any additional CSVs shipped with the dataset can live in the same folder; the
loader will automatically detect them when available.  If you keep the dataset
in a different location, pass `--dataset-root` to the helper scripts.

## Running training

```
python kuairec/train/run.py --dataset-root /path/to/KuaiRec
```

The wrapper sets the environment variables expected by `train/main.py` and
invokes the canonical training loop.  Override hyperparameters by passing the
usual flags (for example `--batch-size 128 --num-epochs 5`).

## Running inference

```
python kuairec/test/run.py \
  --dataset-root /path/to/KuaiRec \
  --checkpoint /path/to/train/ckpt_path/global_stepX.valid_loss=Y/model.pt
```

This script exports the environment variables required by `test/infer.py`,
launches the inference pipeline, and writes all artifacts to the directory
specified by `--result-dir` (defaults to `kuairec/eval_results`).
