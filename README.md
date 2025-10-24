# Generative-AI-for-Ad-Recommendation-using-All-Modality-Historical-Behavior-Data

This project contains a baseline pipeline for generative ad recommendation.  It trains a transformer-style model over multi-modal user histories and produces embeddings that can be searched with FAISS during inference.

## 1. Installation

1. Create a Python 3.9+ environment (virtualenv or Conda).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Prepare a dataset

The data loader detects which dataset is available at `TRAIN_DATA_PATH`:

- **Tencent preprocessed data** – provide the folder that contains `seq.jsonl`, `predict_seq.jsonl`, and the accompanying `*_offsets.pkl` files.
- **KuaiRec** – provide the folder that contains the CSV files (for the light-weight release this is typically `data/small_matrix.csv`, `user_features.csv`, and `item_categories.csv`).  You can download the dataset from [https://kuairec.com/](https://kuairec.com/) and extract it under `data/KuaiRec`.  The loader will look for CSVs in the root you pass as well as in a nested `data/` directory.

> **Tip:** when using KuaiRec, multi-modal embedding IDs (`--mm_emb_id`) are ignored automatically because the dataset does not ship with those features.

### Verify that the files are visible

Run the dataset inspector to confirm that the loader can see your files and to preview a few user sequences:

```bash
python -m train.dataset --data-path /path/to/KuaiRec --sample-users 2 --show-features --watch-ratio-threshold 2.0
```

> Prefer working from inside the `train/` directory? Run `cd train` followed by `python dataset.py --data-path ../data/KuaiRec`
> (or omit `--data-path` if your dataset lives in `../data`). Add `--watch-ratio-threshold 2.0` when you want to keep only the high-engagement KuaiRec interactions (`watch_ratio ≥ 2.0`, which matches the official "like" heuristic).

Example output (numbers depend on the exact split you downloaded):

```
Detected dataset type: kuairec
Users: 1000  Items: 5000  User sequences: 1000

Derived feature groups:
  - user_sparse: 3 feature(s)
  - item_sparse: 5 feature(s)

Sample sequences:
  • User #0: history length=42, next positive item id=87 (original=346892)
  • User #1: history length=37, next positive item id=313 (original=148227)

Use python train/main.py --data_path "/path/to/KuaiRec" --watch_ratio_threshold 2.0 to launch full training once the dataset looks correct. You can
export TRAIN_DATA_PATH instead of passing --data_path each time.
```

If you only want to check that the Python files compile (without actually loading any data), you can run `python -m compileall train test`.  This is a static syntax check and it will print the directories it has scanned—no training or inference happens at this stage.

## 3. Run training

```bash
export TRAIN_DATA_PATH=/path/to/your/dataset/root
# Optional: override where logs and checkpoints are written
export TRAIN_LOG_PATH=./logs
export TRAIN_TF_EVENTS_PATH=./events
export TRAIN_CKPT_PATH=./ckpt_path

# Either rely on TRAIN_DATA_PATH or pass --data_path explicitly
python train/main.py --device cpu --batch_size 128 --maxlen 100 --num_epochs 1 --watch_ratio_threshold 2.0
# or
python train/main.py --device cpu --batch_size 128 --maxlen 100 --num_epochs 1 --data_path /path/to/your/dataset/root \
  --watch_ratio_threshold 2.0
```

During a run you will see messages like:

```
Using dataset root: /path/to/KuaiRec/data
Loaded kuairec dataset with 1000 user sequence(s), 1000 users, 5000 items.
KuaiRec interactions kept for training: 452000 / 460000 rows.
  • Dropped 8000 rows with missing user/item IDs.
  • Filtered out 12000 low-engagement rows (watch_ratio_threshold=2.0).
User sequences contain 452000 chronological interactions after preprocessing.
Training objective: sequential next-item InfoNCE with in-batch negatives (identical to the Tencent pipeline).
Start training
{"global_step": 0, "loss": 0.6932, "epoch": 1, "time": 1.697e09}
...
{"global_step": 120, "loss": 0.5124, "epoch": 1, "time": 1.697e09}
Done
```

Those KuaiRec summaries confirm that the CSV interactions were mapped into the same sequential format as the Tencent pipeline—the
training loop still performs next-item prediction with in-batch negatives via the InfoNCE objective.

Key arguments you may want to tune:

- `--device` – use `cuda` if a GPU is available.
- `--maxlen` – truncate/pad the length of interaction sequences.
- `--mm_emb_id` – only relevant for Tencent data with pre-computed multimedia embeddings.
- `--watch_ratio_threshold` – KuaiRec only; filter interactions by `watch_ratio`. Set this to `2.0` if you want to keep only the most-engaged views (the dataset maintainers' recommendation for synthesising a like signal).
- `--num_epochs`, `--batch_size`, and `--lr` – standard optimisation controls for training duration and learning rate.

The script automatically creates a validation split (90/10) from the training data and logs losses to TensorBoard (`TRAIN_TF_EVENTS_PATH`).

## 4. Run inference / embedding export

1. Set the environment variables:
   ```bash
   export MODEL_OUTPUT_PATH=/path/to/a/folder/with/model.pt
   export EVAL_DATA_PATH=/path/to/eval/dataset/root
   export EVAL_RESULT_PATH=./eval_results
   mkdir -p "$EVAL_RESULT_PATH"
   ```
2. Launch inference:
   ```bash
   python test/infer.py --device cpu --maxlen 100
   ```

`test/infer.py` loads the same dataset format as training, exports user embeddings to `query.fbin`, and (when the FAISS demo binaries are available under `/workspace/faiss-based-ann`) performs approximate nearest-neighbour retrieval.  The script writes its outputs to `EVAL_RESULT_PATH`.

During a successful run the console shows the dataset summary followed by progress bars for embedding export and ANN search.  The resulting binary files (`query.fbin`, `gallery.fbin`) and the human-readable top‑k recommendations (`rank.txt`) are placed under `EVAL_RESULT_PATH`.

## 5. Optional: quantising multi-modal embeddings

The `train/model_rqvae.py` file sketches an RQ-VAE / k-means workflow for compressing external multimedia embeddings into discrete semantic IDs.  This step is optional and only relevant when working with the Tencent dataset's large embedding files.

## 6. Repository structure

```
train/
  main.py        # training loop
  dataset.py     # Tencent + KuaiRec data loading logic
  model.py       # baseline transformer recommendation model
  run.sh         # helper script that launches training

test/
  infer.py       # exports user/item embeddings and runs ANN retrieval
```

Feel free to adapt the hyperparameters and logging directories to match your environment or cloud runner.
