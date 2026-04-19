# LLM-rec/src/build_generative_user_profiles.py
"""
Generate natural-language user interest summaries using a local Ollama
model (default: llama3.1:8b) or HuggingFace Inference API.

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

Usage (Ollama — default, no API key needed):
    ollama serve          # if not already running
    python LLM-rec/src/build_generative_user_profiles.py

Usage (HuggingFace API):
    # set backend: huggingface in generation_config.yaml, then:
    python LLM-rec/src/build_generative_user_profiles.py

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
    pass

import numpy as np
import pandas as pd
from tqdm import tqdm

from generation_utils import load_generation_config, now_iso
from build_item_llm_embeddings import build_text

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "LLM-rec" / "config" / "generation_config.yaml"


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            data = json.load(f)
        n = len(data.get("summaries", {}))
        print(f"  Resuming from cache: {n:,} users already done")
        return data
    return {"model": "", "backend": "", "generated_at": "", "summaries": {}}


def _save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.parent / (cache_path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        tmp.replace(cache_path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ── Profile building ───────────────────────────────────────────────────────────

def _load_metadata(data_dir: Path, item_cfg: dict) -> dict[int, str]:
    meta_path = data_dir / "kuairec_caption_category.csv"
    try:
        df = pd.read_csv(meta_path, sep=None, engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(meta_path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    vid_col = next((c for c in ["video_id", "item_id", "id"] if c in df.columns), None)
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
    sorted_df = big_matrix.sort_values("watch_ratio", ascending=False)
    groups = sorted_df.groupby("user_id")
    user_ids, texts = [], []
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


# ── Generative backends ────────────────────────────────────────────────────────

def _generate_ollama(prompt: str, model: str, host: str,
                     max_tokens: int, temperature: float) -> str:
    """Call local Ollama server."""
    import ollama
    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens, "temperature": temperature},
    )
    return response.message.content.strip()


def _generate_hf(prompt: str, model: str, token: str,
                 max_tokens: int, temperature: float) -> str:
    """Call HuggingFace Inference API."""
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=model, token=token)
    response = client.text_generation(
        prompt, max_new_tokens=max_tokens,
        temperature=temperature, do_sample=temperature > 0,
    )
    text = response.strip()
    if "Summary:" in text:
        text = text.split("Summary:")[-1].strip()
    return text


# ── Main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate user interest summaries via Ollama or HuggingFace."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    cfg = load_generation_config(args.config)
    item_cfg = cfg["item_profile"]
    user_cfg = cfg["user_profile"]
    gen_cfg = cfg.get("generative", {})

    if not gen_cfg.get("enabled", False):
        print("Generative summarization is disabled (generative.enabled: false).")
        sys.exit(0)

    backend = gen_cfg.get("backend", "ollama")
    max_new_tokens = gen_cfg.get("max_new_tokens", 80)
    temperature = gen_cfg.get("temperature", 0.2)
    prompt_template = gen_cfg["prompt_template"]
    cache_path = ROOT / gen_cfg["cache_path"]

    # ── Pick model + validate ──────────────────────────────────────────────────
    if backend == "ollama":
        model_id = gen_cfg.get("ollama_model", "llama3.1:8b")
        ollama_host = gen_cfg.get("ollama_host", "http://localhost:11434")
        try:
            import ollama as _ollama
            _ollama.Client(host=ollama_host).list()   # connection test
            print(f"Ollama connected at {ollama_host}")
        except Exception as e:
            print(f"ERROR: Cannot connect to Ollama at {ollama_host}")
            print(f"  Make sure Ollama is running:  ollama serve")
            print(f"  Detail: {e}")
            sys.exit(1)
        hf_token = None
    else:
        model_id = gen_cfg.get("hf_model", "mistralai/Mistral-7B-Instruct-v0.3")
        token_env = gen_cfg.get("hf_token_env", "HF_TOKEN")
        hf_token = os.environ.get(token_env)
        if not hf_token:
            print(f"ERROR: {token_env} not set. Add it to .env or export it.")
            sys.exit(1)

    print(f"Backend : {backend}")
    print(f"Model   : {model_id}")
    print(f"Params  : max_new_tokens={max_new_tokens}, temperature={temperature}")

    # ── Load data ──────────────────────────────────────────────────────────────
    data_dir = ROOT / "kuairec" / "data"
    print("\nLoading interaction data…")
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

    # ── Cache ──────────────────────────────────────────────────────────────────
    cache = _load_cache(cache_path)
    cache["model"] = model_id
    cache["backend"] = backend
    if not cache.get("generated_at"):
        cache["generated_at"] = now_iso()
    summaries: dict[str, str] = cache.setdefault("summaries", {})

    remaining = [(uid, txt) for uid, txt in zip(user_ids, raw_texts)
                 if str(uid) not in summaries]
    print(f"\n{len(summaries):,} done, {len(remaining):,} remaining.\n")

    if not remaining:
        print("All users already summarised.")
        print(f"Next: python LLM-rec/src/build_user_llm_embeddings.py --use_generative")
        return

    # ── Generate ───────────────────────────────────────────────────────────────
    errors = 0
    rate_limit = gen_cfg.get("rate_limit_seconds", 0) if backend != "ollama" else 0

    for uid, raw_text in tqdm(remaining, desc=f"Generating ({backend})"):
        prompt = prompt_template.format(raw_profile=raw_text)
        try:
            if backend == "ollama":
                summary = _generate_ollama(prompt, model_id, ollama_host,
                                           max_new_tokens, temperature)
            else:
                summary = _generate_hf(prompt, model_id, hf_token,
                                       max_new_tokens, temperature)
        except KeyboardInterrupt:
            print("\n\nInterrupted — progress saved. Re-run to continue.")
            _save_cache(cache, cache_path)
            sys.exit(0)
        except Exception as e:
            errors += 1
            print(f"\n  Warning: failed for user {uid}: {e}")
            summary = raw_text[:200]   # fallback to truncated raw text

        summaries[str(uid)] = summary
        _save_cache(cache, cache_path)
        if rate_limit:
            time.sleep(rate_limit)

    # ── Report ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Done. {len(summaries):,} summaries saved to:")
    print(f"  {cache_path}")
    if errors:
        print(f"  ({errors} errors — truncated raw text used as fallback)")
    print("\nExample summaries:")
    for uid_str, summary in list(summaries.items())[:3]:
        print(f"  User {uid_str}: {summary[:100]}...")
    print(f"\nNext step:")
    print(f"  python LLM-rec/src/build_user_llm_embeddings.py --use_generative")


if __name__ == "__main__":
    main()
