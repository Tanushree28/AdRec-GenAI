# LLM-rec/src/build_item_llm_embeddings.py
"""
Build per-item LLM embeddings from KuaiRec video metadata.

Every generation parameter is read from generation_config.yaml so that
results are fully reproducible. A JSON sidecar is written alongside the
.npy file documenting the exact model, encoding settings, prompt template,
and example outputs used.

Usage:
    python LLM-rec/src/build_item_llm_embeddings.py
    python LLM-rec/src/build_item_llm_embeddings.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from generation_utils import (
    load_generation_config,
    get_model_revision,
    get_st_version,
    get_model_device,
    compute_md5,
    count_tokens,
    save_metadata,
    now_iso,
    embedding_l2_norm,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "LLM-rec" / "config" / "generation_config.yaml"


def build_text(row: pd.Series, fields: list[dict], separator: str, fallback: str) -> str:
    """
    Convert one metadata row into a single text string using the field
    definitions from generation_config.yaml.

    Template (all fields present):
        "cover: <manual_cover_text> | caption: <caption> | topics: <topic_tag> |
         category1: <first_level_category_name> | category2: <second_level_category_name>"
    """
    parts = []
    for field in fields:
        col, prefix = field["column"], field["prefix"]
        if col in row and isinstance(row[col], str) and row[col].strip():
            parts.append(f"{prefix}: {row[col].strip()}")
    return separator.join(parts) if parts else fallback


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build item LLM embeddings for KuaiRec.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to generation_config.yaml (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    cfg = load_generation_config(args.config)
    model_cfg = cfg["model"]
    enc_cfg = cfg["encoding"]
    item_cfg = cfg["item_profile"]
    repro_cfg = cfg["reproducibility"]

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_path = ROOT / "kuairec" / "data" / "kuairec_caption_category.csv"
    out_dir = ROOT / "kuairec" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "item_llm_embeddings.npy"
    meta_path = out_dir / "item_generation_metadata.json"

    # ── Load metadata CSV ──────────────────────────────────────────────────────
    print(f"Loading item metadata from: {data_path}")
    try:
        df = pd.read_csv(data_path, sep=None, engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(data_path, sep=None, engine="python")

    df.columns = [c.strip() for c in df.columns]
    print(f"  Columns: {df.columns.tolist()}")

    vid_col = next((c for c in ["video_id", "item_id", "id"] if c in df.columns), None)
    if vid_col is None:
        raise KeyError(f"No video/item id column found in: {df.columns.tolist()}")

    df[vid_col] = pd.to_numeric(df[vid_col], errors="coerce")
    df = df.dropna(subset=[vid_col])
    df[vid_col] = df[vid_col].astype(int)
    print(f"  {len(df):,} items after id cleaning")

    # ── Build item text descriptions ───────────────────────────────────────────
    print("Building item text descriptions…")
    texts = [
        build_text(row, item_cfg["fields"], item_cfg["field_separator"], item_cfg["fallback_text"])
        for _, row in df.iterrows()
    ]
    n_fallback = sum(1 for t in texts if t == item_cfg["fallback_text"])
    print(f"  {n_fallback} items used fallback text '{item_cfg['fallback_text']}'")

    # ── Load sentence-transformer ──────────────────────────────────────────────
    device_override = enc_cfg.get("device") or None
    print(f"Loading model: {model_cfg['name']} (revision={model_cfg.get('revision')})")
    model = SentenceTransformer(
        model_cfg["name"],
        revision=model_cfg.get("revision") or None,
        device=device_override,
    )
    actual_revision = get_model_revision(model)
    actual_device = get_model_device(model)
    st_version = get_st_version()
    print(f"  sentence-transformers=={st_version}, device={actual_device}")
    if actual_revision:
        print(f"  Model revision: {actual_revision}")
        print("  → Paste this revision into generation_config.yaml model.revision to lock weights.")

    # ── Encode ─────────────────────────────────────────────────────────────────
    batch_size = enc_cfg["batch_size"]
    normalize = enc_cfg["normalize_embeddings"]
    print(f"Encoding {len(texts):,} texts (batch_size={batch_size}, normalize={normalize})…")

    all_embeddings: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embeddings.append(emb)

    all_embeddings_np = np.vstack(all_embeddings).astype(enc_cfg["precision"])
    emb_dim = all_embeddings_np.shape[1]

    # ── Build dense item matrix indexed by video_id ────────────────────────────
    video_ids = df[vid_col].values
    max_id = int(video_ids.max())
    n_items = max_id + 1
    item_emb = np.zeros((n_items, emb_dim), dtype=np.float32)
    for idx, vid in enumerate(video_ids):
        if 0 <= int(vid) < n_items:
            item_emb[int(vid)] = all_embeddings_np[idx]

    np.save(out_path, item_emb)
    print(f"\nSaved item_llm_embeddings.npy — shape {item_emb.shape}")

    # ── Collect example outputs ────────────────────────────────────────────────
    n_examples = repro_cfg.get("n_example_outputs", 5)
    token_counts = count_tokens(texts[:n_examples], model)
    examples = []
    for i in range(min(n_examples, len(df))):
        vid = int(df.iloc[i][vid_col])
        examples.append({
            "item_id": vid,
            "text": texts[i],
            "token_count": token_counts[i] if i < len(token_counts) else None,
            "embedding_l2_norm": embedding_l2_norm(item_emb[vid]) if vid < n_items else None,
        })

    # ── Write metadata sidecar ─────────────────────────────────────────────────
    if repro_cfg.get("save_generation_metadata", True):
        md5 = compute_md5(out_path)
        metadata = {
            "generated_at": now_iso(),
            "config_path": str(args.config),
            "model": {
                "name": model_cfg["name"],
                "sentence_transformers_version": st_version,
                "revision": actual_revision,
                "pooling_strategy": model_cfg.get("pooling_strategy", "mean"),
                "max_seq_length": model_cfg.get("max_seq_length", 128),
                "embedding_dim": emb_dim,
            },
            "encoding": {
                "batch_size": batch_size,
                "normalize_embeddings": normalize,
                "device": actual_device,
                "precision": enc_cfg["precision"],
            },
            "item_profile": {
                "field_separator": item_cfg["field_separator"],
                "fallback_text": item_cfg["fallback_text"],
                "fields": [f["prefix"] for f in item_cfg["fields"]],
                "prompt_template": item_cfg["field_separator"].join(
                    f"{f['prefix']}: <{f['column']}>" for f in item_cfg["fields"]
                ),
                "n_items_total": len(df),
                "n_fallback_used": n_fallback,
            },
            "output": {
                "path": str(out_path),
                "shape": list(item_emb.shape),
                "dtype": str(item_emb.dtype),
                "md5": md5,
            },
            "examples": examples,
        }
        save_metadata(metadata, meta_path)
        print(f"Metadata sidecar written to: {meta_path}")
        print(f"  MD5: {md5}")


if __name__ == "__main__":
    main()
