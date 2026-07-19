# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Generative AI for ad recommendation using multi-modal historical behavior data. There are two distinct training pipelines:

1. **Baseline transformer pipeline** (`train/`, `test/`) — SASRec-style sequential recommendation model trained with InfoNCE loss, supports both Tencent and KuaiRec datasets.
2. **LLM-enhanced pipeline** (`LLM-rec/`) — MLP models that optionally incorporate sentence-transformer item embeddings for watch-ratio prediction on KuaiRec.

## Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A `.venv` directory is already present at the project root.

## Commands

### Baseline Transformer Pipeline

**Training (Tencent or KuaiRec):**
```bash
export TRAIN_DATA_PATH=/path/to/dataset
export TRAIN_LOG_PATH=./logs
export TRAIN_TF_EVENTS_PATH=./events
export TRAIN_CKPT_PATH=./ckpt_path

python train/main.py --device cpu --batch_size 128 --maxlen 100
```

**KuaiRec convenience wrapper (sets env vars automatically):**
```bash
python kuairec/train/run.py --dataset-root /path/to/KuaiRec
```

**Inference / embedding export:**
```bash
export MODEL_OUTPUT_PATH=/path/to/ckpt_path/global_stepX.valid_loss=Y
export EVAL_DATA_PATH=/path/to/dataset
export EVAL_RESULT_PATH=./eval_results

python test/infer.py --device cpu --maxlen 100
# KuaiRec wrapper:
python kuairec/test/run.py --dataset-root /path/to/KuaiRec --checkpoint /path/to/model.pt
```

**Inspect inference artifacts:**
```bash
python tools/dump_eval_results.py ./eval_results --sample-users 3 --write-json preview.json
```

### LLM-Enhanced Pipeline

**Build sentence-transformer item embeddings (run once before LLM training):**
```bash
python LLM-rec/src/build_item_llm_embeddings.py
# Reads: kuairec/data/kuairec_caption_category.csv
# Writes: kuairec/embeddings/item_llm_embeddings.npy
```

**Train baseline MLP:**
```bash
cd LLM-rec/src && python train_baseline.py
```

**Train LLM-enhanced MLP:**
```bash
cd LLM-rec/src && python train_llm.py
```

Config for both is at `LLM-rec/config/base.yaml` — edit data paths and hyperparameters there.

### Testing and Linting

```bash
pytest                     # run all tests
pytest test/               # run inference-side tests only
black .                    # format code
```

## Architecture

### Baseline Transformer Model (`train/model.py`)

`BaselineModel` is a transformer encoder (SASRec-style) that embeds user/item IDs plus sparse categorical features, runs multi-head self-attention over the interaction sequence, and returns contextualized sequence representations. Training uses InfoNCE loss (`compute_infonce_loss`) over positive/negative item pairs. Checkpoints are saved per epoch as `global_stepN.valid_loss=X/model.pt`.

### Dataset Loading (`train/dataset.py`, `kuairec/train/dataset.py`)

`MyDataset` auto-detects the dataset type from the data path:
- **Tencent**: expects `seq.jsonl` + `*_offsets.pkl` offset files for random-access streaming.
- **KuaiRec**: expects `small_matrix.csv` (or `big_matrix.csv`), optionally `user_features.csv` and `item_categories.csv`. Multi-modal embedding IDs (`--mm_emb_id`) are silently ignored for KuaiRec.

### LLM-Enhanced Models (`LLM-rec/src/models/`)

- `BaselineMLP` — collaborative filtering MLP, takes user/item ID embeddings, predicts watch ratio.
- `LLMEnhancedMLP` — extends `BaselineMLP` by concatenating pre-computed sentence-transformer embeddings (dim 384 by default) to the item representation before the MLP layers.

### Inference Pipeline (`test/infer.py`)

Exports `query.fbin` (user embeddings) and `embedding.fbin` + `id.u64bin` (item embeddings). If FAISS ANN binaries are available under `/workspace/faiss-based-ann`, it runs top-k retrieval and writes `id100.u64bin`.

### KuaiRec Wrappers (`kuairec/`)

Self-contained mirror of `train/` and `test/` with dataset-specific defaults. `kuairec/train/run.py` and `kuairec/test/run.py` set the required environment variables and invoke `kuairec.train.main` / `kuairec.test.main` as modules. These are the recommended entry points when working with KuaiRec data.
