# LLM-rec/src/build_generative_user_profiles.py
"""
Generate natural-language user interest summaries using a HuggingFace
generative model (Mistral-7B-Instruct-v0.3 by default).

This is the "Generative AI" step that justifies the paper title. For each
user we feed their raw top-K item concatenation into the model with an
instruction prompt, and it writes a 2-sentence plain-English summary of
their interests. These summaries are then encoded by the sentence-transformer
in build_user_llm_embeddings.py --use_generative.

Why this improves on raw concatenation:
  - Raw profiles average 583 tokens but the encoder truncates at 128
    (~78% of the profile is silently discarded).
  - A 2-sentence summary fits within 128 tokens, so the encoder reads
    the entire profile.

Usage:
    export HF_TOKEN=hf_xxxx
    python LLM-rec/src/build_generative_user_profiles.py
    python LLM-rec/src/build_generative_user_profiles.py --config path/to/config.yaml

The script is fully resumable: if interrupted, re-run and it skips users
already saved in the cache JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Auto-load .env from project root if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass  # python-dotenv not installed — rely on env var being set manually

import numpy as np
import pandas as pd
from tqdm import tqdm

from generation_utils import load_generation_config, now_iso, save_metadata
from build_item_llm_embeddings import build_text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "LLM-rec" / "config" / "generation_config.yaml"


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache(cache_path: Path) -> dict:
    """Load existing cache or return empty structure."""
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Resuming from cache: {len(data.get('summaries', {}))} users already done")
        return data
    return {"model": "", "generated_at": "", "summaries": {}}


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Atomically write cache to disk after every user (safe to interrupt)."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.parent / (cache_path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        tmp.replace(cache_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ── Profile building (reused from build_user_llm_embeddings) ──────────────────

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


def _build_raw_profiles(
    big_matrix: pd.DataFrame,
    id2text: dict[int, str],
    top_k: int,
    item_separator: str,
    deduplicate: bool,
    fallback_text: str,
) -> tuple[list[int], list[str]]:
    """Return (user_ids, raw_profile_texts) — same logic as build_user_llm_embeddings
    but WITHOUT the prompt prefix, so the generative model gets clean input."""
    sorted_df = big_matrix.sort_values("watch_ratio", ascending=False)
    groups = sorted_df.groupby("user_id")
    user_ids: list[int] = []
    texts: list[str] = []
    for uid, grp in tqdm(groups, desc="Building raw profiles"):
        top_items = grp["video_id"].head(top_k).tolist()
        descriptions = [id2text.get(int(vid), fallback_text) for vid in top_items]
        if deduplicate:
            seen: set[str] = set()
            unique: list[str] = []
            for d in descriptions:
                if d not in seen:
                    seen.add(d)
                    unique.append(d)
            descriptions = unique
        user_ids.append(int(uid))
        texts.append(item_separator.join(descriptions))
    return user_ids, texts


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate user interest summaries via HuggingFace generative model."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Path to generation_config.yaml (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args(argv)

    cfg = load_generation_config(args.config)
    item_cfg = cfg["item_profile"]
    user_cfg = cfg["user_profile"]
    gen_cfg = cfg.get("generative", {})

    if not gen_cfg.get("enabled", False):
        print("Generative summarization is disabled in config (generative.enabled: false).")
        print("Set enabled: true in generation_config.yaml to use this script.")
        sys.exit(0)

    # ── HF token ──────────────────────────────────────────────────────────────
    token_env = gen_cfg.get("hf_token_env", "HF_TOKEN")
    hf_token = os.environ.get(token_env)
    if not hf_token:
        print(f"ERROR: {token_env} environment variable not set.")
        print(f"  Run:  export {token_env}=hf_xxxx")
        sys.exit(1)

    model_id = gen_cfg["model"]
    max_new_tokens = gen_cfg.get("max_new_tokens", 80)
    temperature = gen_cfg.get("temperature", 0.2)
    prompt_template = gen_cfg["prompt_template"]
    rate_limit = gen_cfg.get("rate_limit_seconds", 0.5)

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir = ROOT / "kuairec" / "data"
    cache_path = ROOT / gen_cfg["cache_path"]

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading big_matrix…")
    big_matrix = pd.read_csv(
        data_dir / "big_matrix.csv",
        dtype={"user_id": "int32", "video_id": "int32"},
        usecols=["user_id", "video_id", "watch_ratio"],
    )
    print(f"  {big_matrix['user_id'].nunique():,} users")

    print("Loading item metadata…")
    id2text = _load_metadata(data_dir, item_cfg)

    print(f"\nBuilding raw user profiles (top_k={user_cfg['top_k']})…")
    user_ids, raw_texts = _build_raw_profiles(
        big_matrix, id2text,
        top_k=user_cfg["top_k"],
        item_separator=user_cfg["item_separator"],
        deduplicate=user_cfg["deduplicate_descriptions"],
        fallback_text=item_cfg["fallback_text"],
    )

    # ── Load or create cache ───────────────────────────────────────────────────
    cache = _load_cache(cache_path)
    cache["model"] = model_id
    if not cache.get("generated_at"):
        cache["generated_at"] = now_iso()
    summaries: dict[str, str] = cache.setdefault("summaries", {})

    already_done = len(summaries)
    remaining = [(uid, txt) for uid, txt in zip(user_ids, raw_texts)
                 if str(uid) not in summaries]
    print(f"\n{already_done} users already summarised, {len(remaining)} remaining.")

    if not remaining:
        print("All users already summarised — nothing to do.")
        print(f"Cache: {cache_path}")
        return

    # ── Load HF InferenceClient ────────────────────────────────────────────────
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("ERROR: huggingface_hub not installed.")
        print("  Run:  pip install huggingface_hub>=0.20")
        sys.exit(1)

    print(f"\nConnecting to HuggingFace Inference API…")
    print(f"  Model: {model_id}")
    print(f"  max_new_tokens={max_new_tokens}, temperature={temperature}")
    print(f"  Rate limit: {rate_limit}s between calls")
    client = InferenceClient(model=model_id, token=hf_token)

    # ── Generate summaries ─────────────────────────────────────────────────────
    errors = 0
    print(f"\nGenerating summaries… (Ctrl+C to pause — progress is saved after each user)\n")

    for uid, raw_text in tqdm(remaining, desc="Generating"):
        prompt = prompt_template.format(raw_profile=raw_text)
        try:
            response = client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            summary = response.strip()
            # Strip any repeated prompt artifacts
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()
        except KeyboardInterrupt:
            print("\n\nInterrupted — progress saved. Re-run to continue.")
            _save_cache(cache, cache_path)
            sys.exit(0)
        except Exception as e:
            errors += 1
            print(f"\n  Warning: failed for user {uid}: {e}")
            # Fall back to the raw profile prefix as the summary
            summary = raw_text[:200]

        summaries[str(uid)] = summary
        _save_cache(cache, cache_path)
        time.sleep(rate_limit)

    # ── Final report ───────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Done. {len(summaries):,} summaries saved to:")
    print(f"  {cache_path}")
    if errors:
        print(f"  ({errors} errors — fallback text used for those users)")
    print(f"\nExample summaries:")
    for uid_str, summary in list(summaries.items())[:3]:
        print(f"  User {uid_str}: {summary[:120]}...")
    print(f"\nNext step:")
    print(f"  python LLM-rec/src/build_user_llm_embeddings.py --use_generative")


if __name__ == "__main__":
    main()
