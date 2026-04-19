# LLM-rec/src/generation_utils.py
"""
Shared utilities for LLM embedding generation reproducibility.

Used by:
  - build_item_llm_embeddings.py
  - build_user_llm_embeddings.py
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Config loading ─────────────────────────────────────────────────────────────

def load_generation_config(path: str | Path) -> dict:
    """Load and lightly validate the generation config YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Generation config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Minimal structural checks
    for section in ("model", "encoding", "item_profile", "user_profile", "reproducibility"):
        if section not in cfg:
            raise KeyError(f"Missing section '{section}' in {path}")

    return cfg


# ── Model introspection ────────────────────────────────────────────────────────

def get_model_revision(model) -> str | None:
    """
    Extract the HuggingFace commit hash from a loaded SentenceTransformer.

    Returns the hash string if it can be found in the local model cache,
    or None if the model was loaded from a non-HF source.
    """
    try:
        # SentenceTransformer stores the model path in ._model_card_vars or
        # the underlying AutoModel's config._name_or_path
        base = getattr(model, "_model_card_vars", {})
        if "model_id" in base:
            return base["model_id"]

        # Try reading refs/main from the HuggingFace local cache
        transformer = getattr(model, "_first_module", None) or model
        config = getattr(getattr(transformer, "auto_model", None), "config", None)
        name_or_path: str | None = getattr(config, "_name_or_path", None)
        if name_or_path and os.path.isdir(name_or_path):
            refs_path = Path(name_or_path).parent.parent / "refs" / "main"
            if refs_path.exists():
                return refs_path.read_text().strip()
    except Exception:
        pass
    return None


def get_st_version() -> str:
    """Return installed sentence-transformers version string."""
    try:
        import sentence_transformers
        return sentence_transformers.__version__
    except Exception:
        return "unknown"


def get_model_device(model) -> str:
    """Return the device the SentenceTransformer is running on (e.g. 'cpu', 'cuda:0')."""
    try:
        return str(next(model.parameters()).device)
    except Exception:
        return "unknown"


# ── File checksums ─────────────────────────────────────────────────────────────

def compute_md5(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return MD5 hex digest of a file (streaming, works on large .npy files)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# ── Token counting ─────────────────────────────────────────────────────────────

def count_tokens(texts: list[str], model) -> list[int]:
    """
    Return the number of tokens for each text string using the model's tokenizer.
    Counts are BEFORE truncation, so values may exceed max_seq_length.
    """
    try:
        tokenizer = model.tokenizer
        return [len(tokenizer.encode(t, add_special_tokens=True)) for t in texts]
    except Exception:
        return []


# ── Metadata persistence ───────────────────────────────────────────────────────

def save_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    """
    Atomically write metadata dict as a pretty-printed JSON file.

    Uses a temp file + rename so the output is never partially written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Generative summary cache ───────────────────────────────────────────────────

def load_generative_summaries(path: str | Path) -> dict[int, str]:
    """
    Load {user_id (int): summary_text (str)} from the generative cache JSON.

    The cache is written by build_generative_user_profiles.py and has the form:
        {
          "model": "mistralai/Mistral-7B-Instruct-v0.3",
          "generated_at": "...",
          "summaries": {"0": "This user enjoys...", "1": "...", ...}
        }

    Raises FileNotFoundError with a helpful message if the cache doesn't exist yet.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Generative summaries cache not found: {path}\n"
            "Run build_generative_user_profiles.py first:\n"
            "  export HF_TOKEN=hf_xxxx\n"
            "  python LLM-rec/src/build_generative_user_profiles.py"
        )
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.get("summaries", {}).items()}


# ── Embedding statistics ───────────────────────────────────────────────────────

def embedding_l2_norm(vec: np.ndarray) -> float:
    """L2 norm of a single embedding vector, rounded to 6 decimal places."""
    return round(float(np.linalg.norm(vec)), 6)
