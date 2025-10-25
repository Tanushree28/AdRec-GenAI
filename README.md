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

- **Tencent preprocessed data** – provide the folder that contains `seq.jsonl` (always required) and the accompanying `*_offsets.pkl` files.  If `predict_seq.jsonl`/`predict_seq_offsets.pkl` are also present they will be used during inference; otherwise the loader falls back to the training sequences.
- **KuaiRec** – provide the folder that contains the CSV files (for the light-weight release this is typically `data/small_matrix.csv`, `user_features.csv`, and `item_categories.csv`).  You can download the dataset from [https://kuairec.com/](https://kuairec.com/) and extract it under `data/KuaiRec`.  The loader will look for CSVs in the root you pass as well as in a nested `data/` directory.

> **Tip:** when using KuaiRec, multi-modal embedding IDs (`--mm_emb_id`) are ignored automatically because the dataset does not ship with those features.

## 3. Run training

```bash
export TRAIN_DATA_PATH=/path/to/your/dataset/root
# Optional: override where logs and checkpoints are written
export TRAIN_LOG_PATH=./logs
export TRAIN_TF_EVENTS_PATH=./events
export TRAIN_CKPT_PATH=./ckpt_path

python train/main.py --device cpu --batch_size 128 --maxlen 100
```

Key arguments you may want to tune:

- `--device` – use `cuda` if a GPU is available.
- `--maxlen` – truncate/pad the length of interaction sequences.
- `--mm_emb_id` – only relevant for Tencent data with pre-computed multimedia embeddings.

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
