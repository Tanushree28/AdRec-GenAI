# LLM-rec/src/build_user_llm_embeddings.py
"""
Build per-user LLM embeddings from KuaiRec interaction history.

For each user we collect their top-K watched videos (by watch_ratio),
build a natural-language interest profile from the item metadata, and
encode it with the same sentence-transformer used for item embeddings.

Every generation parameter is read from generation_config.yaml.
A JSON sidecar is written alongside the .npy file recording the exact
model, encoding settings, prompt template, population statistics,
and example user profiles for reproducibility.

Usage:
    python LLM-rec/src/build_user_llm_embeddings.py
    python LLM-rec/src/build_user_llm_embeddings.py --config path/to/config.yaml

Full prompt example (top_k=3, deduplicate=True):
    "This user frequently watches: cover: 旅行 | caption: 海边日落 | topics: 旅游;
     caption: 美食推荐 | category1: 美食; topics: 音乐 | category1: 娱乐"
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
from build_item_llm_embeddings import build_text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "LLM-rec" / "config" / "generation_config.yaml"


def _load_metadata(data_dir: Path, item_cfg: dict) -> dict[int, str]:
    """Return {video_id: item_text} from the caption/category CSV."""
    meta_path = data_dir / "kuairec_caption_category.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    try:
        df = pd.read_csv(meta_path, sep=None, engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(meta_path, sep=None, engine="python")

    df.columns = [c.strip() for c in df.columns]
    vid_col = next((c for c in ["video_id", "item_id", "id"] if c in df.columns), None)
    if vid_col is None:
        raise KeyError(f"No video/item id column found in: {df.columns.tolist()}")

    df[vid_col] = pd.to_numeric(df[vid_col], errors="coerce")
    df = df.dropna(subset=[vid_col])
    df[vid_col] = df[vid_col].astype(int)

    return {
        int(row[vid_col]): build_text(
            row, item_cfg["fields"], item_cfg["field_separator"], item_cfg["fallback_text"]
        )
        for _, row in df.iterrows()
    }


def _build_user_profiles(
    big_matrix: pd.DataFrame,
    id2text: dict[int, str],
    top_k: int,
    prompt_prefix: str,
    item_separator: str,
    deduplicate: bool,
    fallback_text: str,
) -> tuple[list[int], list[str], dict]:
    """
    Build (user_ids, profile_texts) and population statistics.

    Profile construction per user:
      1. Sort interactions by watch_ratio descending.
      2. Take top-K video IDs.
      3. Map each to its item text description.
      4. Optionally deduplicate identical descriptions.
      5. Join with item_separator and prepend prompt_prefix.
    """
    sorted_df = big_matrix.sort_values("watch_ratio", ascending=False)
    groups = sorted_df.groupby("user_id")

    user_ids: list[int] = []
    texts: list[str] = []
    n_items_used_list: list[int] = []
    n_below_k = 0

    for uid, grp in tqdm(groups, desc="Building user profiles"):
        top_items = grp["video_id"].head(top_k).tolist()

        if len(top_items) < top_k:
            n_below_k += 1

        descriptions = [id2text.get(int(vid), fallback_text) for vid in top_items]

        if deduplicate:
            seen: set[str] = set()
            unique: list[str] = []
            for d in descriptions:
                if d not in seen:
                    seen.add(d)
                    unique.append(d)
            descriptions = unique

        profile = prompt_prefix + item_separator.join(descriptions)
        user_ids.append(int(uid))
        texts.append(profile)
        n_items_used_list.append(len(descriptions))

    stats = {
        "n_users_total": len(user_ids),
        "n_users_below_k": n_below_k,
        "avg_unique_descriptions_per_user": round(
            float(np.mean(n_items_used_list)), 2
        ) if n_items_used_list else 0.0,
        "min_descriptions": int(min(n_items_used_list)) if n_items_used_list else 0,
        "max_descriptions": int(max(n_items_used_list)) if n_items_used_list else 0,
    }
    return user_ids, texts, stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build user LLM profile embeddings for KuaiRec.")
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
    user_cfg = cfg["user_profile"]
    repro_cfg = cfg["reproducibility"]

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir = ROOT / "kuairec" / "data"
    big_matrix_path = data_dir / "big_matrix.csv"
    out_dir = ROOT / "kuairec" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "user_llm_embeddings.npy"
    meta_path = out_dir / "user_generation_metadata.json"

    # ── Load interaction data ──────────────────────────────────────────────────
    print(f"Loading big_matrix from: {big_matrix_path}")
    big_matrix = pd.read_csv(
        big_matrix_path,
        dtype={"user_id": "int32", "video_id": "int32"},
        usecols=["user_id", "video_id", "watch_ratio"],
    )
    print(f"  {len(big_matrix):,} interactions, {big_matrix['user_id'].nunique():,} users")

    # ── Load item metadata ─────────────────────────────────────────────────────
    print("Loading item metadata…")
    id2text = _load_metadata(data_dir, item_cfg)
    print(f"  {len(id2text):,} items with metadata")

    # ── Build user profiles ────────────────────────────────────────────────────
    top_k = user_cfg["top_k"]
    prompt_prefix = user_cfg["prompt_prefix"]
    item_separator = user_cfg["item_separator"]
    deduplicate = user_cfg["deduplicate_descriptions"]

    print(f"\nBuilding user profiles (top_k={top_k}, deduplicate={deduplicate})…")
    print(f"  Prompt prefix: \"{prompt_prefix}\"")
    print(f"  Item separator: \"{item_separator}\"")
    user_ids, texts, pop_stats = _build_user_profiles(
        big_matrix, id2text, top_k, prompt_prefix, item_separator,
        deduplicate, item_cfg["fallback_text"],
    )
    print(f"  {pop_stats['n_users_below_k']} users had fewer than {top_k} interactions")
    print(f"  Avg unique descriptions/user: {pop_stats['avg_unique_descriptions_per_user']}")

    # ── Load sentence-transformer ──────────────────────────────────────────────
    device_override = enc_cfg.get("device") or None
    print(f"\nLoading model: {model_cfg['name']} (revision={model_cfg.get('revision')})")
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
        print("  → Paste this into generation_config.yaml model.revision to lock weights.")

    # ── Encode ─────────────────────────────────────────────────────────────────
    batch_size = enc_cfg["batch_size"]
    normalize = enc_cfg["normalize_embeddings"]
    print(f"\nEncoding {len(texts):,} user profiles (batch_size={batch_size}, normalize={normalize})…")

    all_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        emb = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_embs.append(emb)

    emb_matrix = np.vstack(all_embs).astype(enc_cfg["precision"])
    emb_dim = emb_matrix.shape[1]

    # ── Build dense array indexed by user_id ──────────────────────────────────
    max_uid = max(user_ids)
    dense = np.zeros((max_uid + 1, emb_dim), dtype=np.float32)
    for idx, uid in enumerate(user_ids):
        dense[uid] = emb_matrix[idx]

    np.save(out_path, dense)
    print(f"\nSaved user_llm_embeddings.npy — shape {dense.shape}")

    # ── Collect example outputs ────────────────────────────────────────────────
    n_examples = repro_cfg.get("n_example_outputs", 5)
    example_texts = [texts[i] for i in range(min(n_examples, len(texts)))]
    token_counts = count_tokens(example_texts, model)

    examples = []
    for i in range(min(n_examples, len(user_ids))):
        uid = user_ids[i]
        examples.append({
            "user_id": uid,
            "profile_text": texts[i],
            "profile_token_count": token_counts[i] if i < len(token_counts) else None,
            "embedding_l2_norm": embedding_l2_norm(dense[uid]),
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
            "user_profile": {
                "top_k": top_k,
                "prompt_prefix": prompt_prefix,
                "item_separator": item_separator,
                "deduplicate_descriptions": deduplicate,
                "prompt_template": (
                    f'"{prompt_prefix}<item_1_text>{item_separator}'
                    f'<item_2_text>{item_separator}...{item_separator}<item_{top_k}_text>"'
                ),
                **pop_stats,
            },
            "output": {
                "path": str(out_path),
                "shape": list(dense.shape),
                "dtype": str(dense.dtype),
                "md5": md5,
            },
            "examples": examples,
        }
        save_metadata(metadata, meta_path)
        print(f"Metadata sidecar written to: {meta_path}")
        print(f"  MD5: {md5}")


if __name__ == "__main__":
    main()
