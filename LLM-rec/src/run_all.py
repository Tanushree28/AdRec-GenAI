# LLM-rec/src/run_all.py
"""
Unified training runner for all 4 models in the ablation study.

Run order:
  A1 — Baseline MLP (ID-only)
  A2 — ID-Transformer  (kuairec.train.main, no --use_llm_emb)
  B1 — LLM-MLP        (ID + item LLM emb + user LLM emb)
  B2 — LLM-Transformer (kuairec.train.main --use_llm_emb --use_user_llm)

Each model's metrics are written to LLM-rec/outputs/metrics/<model_key>.json.
A summary table is printed at the end.

Usage (from the project root):
    python -m LLM-rec.src.run_all
    # or run with custom config:
    python -m LLM-rec.src.run_all --config LLM-rec/config/base.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parents[2]
LLM_REC_ROOT = ROOT / "LLM-rec"
DEFAULT_CONFIG = LLM_REC_ROOT / "config" / "base.yaml"
DEFAULT_OUT_DIR = LLM_REC_ROOT / "outputs"
EMB_DIR = ROOT / "kuairec" / "embeddings"


# ---------------------------------------------------------------------------
# Helpers shared by MLP models
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = n = 0
    for batch in loader:
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)
        y = batch["watch_ratio"].to(device)
        optimizer.zero_grad()
        y_hat = model(user_ids, item_ids)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n


def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = n = 0
    with torch.no_grad():
        for batch in loader:
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            y = batch["watch_ratio"].to(device)
            y_hat = model(user_ids, item_ids)
            loss = criterion(y_hat, y)
            total_loss += loss.item() * len(y)
            n += len(y)
    return total_loss / n


def _run_mlp(
    model,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    device,
    ckpt_path: Path,
    metrics_path: Path,
    patience: int = 5,
) -> list[dict]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        t_ep = time.time()
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = _eval_epoch(model, val_loader, criterion, device)
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "epoch_time_sec": time.time() - t_ep}
        history.append(rec)
        print(json.dumps(rec))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(best val_loss {best_val:.6f} @ epoch {best_epoch})")
                break
    print(f"Total: {time.time() - t0:.1f}s | best val_loss {best_val:.6f} @ epoch {best_epoch}")
    with open(metrics_path, "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_loss": best_val,
                   "history": history}, f, indent=2)
    return history


def _pick_device(cfg: dict) -> torch.device:
    """Device priority: config override > cuda > mps (Apple Silicon) > cpu."""
    override = cfg.get("training", {}).get("device")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-model runners
# ---------------------------------------------------------------------------

def run_a1_baseline_mlp(cfg: dict, out_dir: Path) -> list[dict]:
    """Model A1: Baseline MLP (ID-only)."""
    print("\n" + "=" * 60)
    print("Model A1: Baseline MLP (ID-only)")
    print("=" * 60)

    import sys as _sys
    _sys.path.insert(0, str(LLM_REC_ROOT / "src"))
    from data_loader import load_big_matrix, make_dataloaders
    from models.baseline_mlp import BaselineMLP

    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]
    device = _pick_device(cfg)
    df = load_big_matrix(data_cfg["big_matrix_path"], sep=data_cfg["sep"])
    train_loader, val_loader, stats = make_dataloaders(df, train_cfg["batch_size"], train_cfg["val_ratio"])

    model = BaselineMLP(
        n_users=stats["n_users"], n_items=stats["n_items"],
        user_embedding_dim=model_cfg["user_embedding_dim"],
        item_embedding_dim=model_cfg["item_embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
    ).to(device)

    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return _run_mlp(
        model, train_loader, val_loader,
        train_cfg["num_epochs"], train_cfg["learning_rate"], device,
        out_dir / "checkpoints" / "a1_baseline_mlp.pt",
        out_dir / "metrics" / "a1_baseline_mlp.json",
        patience=train_cfg.get("patience", 5),
    )


def run_b1_llm_mlp(cfg: dict, out_dir: Path, generative: bool = False) -> list[dict]:
    """Model B1 / B1-Gen: LLM-MLP (ID + item LLM emb + user LLM emb)."""
    variant = "B1-Gen" if generative else "B1"
    user_emb_file = "user_llm_embeddings_generative.npy" if generative else "user_llm_embeddings.npy"
    print("\n" + "=" * 60)
    print(f"Model {variant}: LLM-MLP (ID + item LLM + {'generative' if generative else 'raw'} user LLM)")
    print("=" * 60)

    import sys as _sys
    _sys.path.insert(0, str(LLM_REC_ROOT / "src"))
    from data_loader import load_big_matrix, make_dataloaders
    from llm_features import load_item_llm_embeddings
    from models.llm_mlp import LLMEnhancedMLP

    data_cfg, train_cfg, model_cfg = cfg["data"], cfg["training"], cfg["model"]
    device = _pick_device(cfg)
    df = load_big_matrix(data_cfg["big_matrix_path"], sep=data_cfg["sep"])
    train_loader, val_loader, stats = make_dataloaders(df, train_cfg["batch_size"], train_cfg["val_ratio"])
    n_items = stats["n_items"]

    # Load item LLM embeddings
    item_llm_path = EMB_DIR / "item_llm_embeddings.npy"
    item_llm = load_item_llm_embeddings(str(item_llm_path))
    if item_llm.shape[0] < n_items:
        pad = torch.zeros(n_items - item_llm.shape[0], item_llm.shape[1])
        item_llm = torch.cat([item_llm, pad], dim=0)
    else:
        item_llm = item_llm[:n_items]

    # Load user LLM embeddings (raw or generative)
    user_llm = None
    user_llm_path = EMB_DIR / user_emb_file
    if user_llm_path.exists():
        user_llm = load_item_llm_embeddings(str(user_llm_path))
        print(f"  Loaded user LLM embeddings ({user_emb_file}): {user_llm.shape}")
    else:
        print(f"  {user_emb_file} not found — running item-LLM-only variant")

    model = LLMEnhancedMLP(
        n_users=stats["n_users"], n_items=n_items,
        user_embedding_dim=model_cfg["user_embedding_dim"],
        item_embedding_dim=model_cfg["item_embedding_dim"],
        llm_embedding_dim=model_cfg["llm_embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        item_llm_embeddings=item_llm,
        user_llm_embeddings=user_llm,
    ).to(device)

    key = "b1_gen_llm_mlp" if generative else "b1_llm_mlp"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return _run_mlp(
        model, train_loader, val_loader,
        train_cfg["num_epochs"], train_cfg["learning_rate"], device,
        out_dir / "checkpoints" / f"{key}.pt",
        out_dir / "metrics" / f"{key}.json",
        patience=train_cfg.get("patience", 5),
    )


def run_transformer(
    model_key: str,
    use_llm: bool,
    use_user_llm: bool,
    cfg: dict,
    out_dir: Path,
    user_llm_path: str | None = None,
) -> None:
    """Run a Transformer model via kuairec.train.main (subprocess)."""
    if "gen" in model_key:
        label = "B2-Gen: LLM-Transformer (generative user emb)"
    elif use_llm:
        label = "B2: LLM-Transformer"
    else:
        label = "A2: ID-Transformer"
    print("\n" + "=" * 60)
    print(f"Model {label}")
    print("=" * 60)

    train_cfg = cfg["training"]
    ckpt_dir = out_dir / "checkpoints" / model_key
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Transformers need a much smaller batch size than MLPs —
    # attention is O(n²) in memory. Fall back to 64 if not set.
    tf_batch = train_cfg.get("transformer_batch_size", 64)
    tf_epochs = train_cfg.get("transformer_num_epochs", train_cfg["num_epochs"])
    tf_device = train_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    cmd = [
        sys.executable, "-m", "kuairec.train.main",
        "--num_epochs", str(tf_epochs),
        "--batch_size", str(tf_batch),
        "--device", tf_device,
    ]
    if use_llm:
        cmd.append("--use_llm_emb")
    if use_user_llm:
        cmd.append("--use_user_llm")
    if user_llm_path:
        cmd += ["--user_llm_path", user_llm_path]

    env_override = {
        "TRAIN_CKPT_PATH": str(ckpt_dir),
        "TRAIN_TF_EVENTS_PATH": str(out_dir / "events" / model_key),
        "TRAIN_LOG_PATH": str(out_dir / "logs" / model_key),
        # small|big — which KuaiRec interaction matrix the transformer trains on
        "KUAIREC_MATRIX": cfg.get("data", {}).get("matrix", "big"),
    }
    import os
    env = {**os.environ, **env_override}

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _best_val(history: list[dict]) -> float:
    return min(r["val_loss"] for r in history)


def print_summary(results: dict[str, list[dict]]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY — Best Validation MSE (MLP models)")
    print("=" * 60)
    print(f"{'Model':<35} {'Best val_loss':>14}")
    print("-" * 52)
    for name, hist in results.items():
        if hist:
            print(f"{name:<35} {_best_val(hist):>14.6f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train all 4 ablation models for the IEEE 2026 paper.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["a1", "a2", "b1", "b2", "b1_gen", "b2_gen", "all"],
        default=["all"],
        help="Which models to train (default: all). b1_gen/b2_gen use generative user embeddings.",
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = DEFAULT_OUT_DIR
    run_all = "all" in args.models
    results: dict[str, list[dict]] = {}

    gen_user_emb = str(EMB_DIR / "user_llm_embeddings_generative.npy")

    if run_all or "a1" in args.models:
        results["A1 Baseline MLP"] = run_a1_baseline_mlp(cfg, out_dir)

    if run_all or "b1" in args.models:
        results["B1 LLM-MLP (raw)"] = run_b1_llm_mlp(cfg, out_dir, generative=False)

    if "b1_gen" in args.models:
        results["B1-Gen LLM-MLP (generative)"] = run_b1_llm_mlp(cfg, out_dir, generative=True)

    if run_all or "a2" in args.models:
        run_transformer("a2_id_transformer", use_llm=False, use_user_llm=False, cfg=cfg, out_dir=out_dir)
        results["A2 ID-Transformer"] = []  # metrics written by kuairec.train.main

    if run_all or "b2" in args.models:
        run_transformer("b2_llm_transformer", use_llm=True, use_user_llm=True, cfg=cfg, out_dir=out_dir)
        results["B2 LLM-Transformer (raw)"] = []  # metrics written by kuairec.train.main

    if "b2_gen" in args.models:
        run_transformer(
            "b2_gen_llm_transformer", use_llm=True, use_user_llm=True,
            cfg=cfg, out_dir=out_dir, user_llm_path=gen_user_emb,
        )
        results["B2-Gen LLM-Transformer (generative)"] = []  # metrics written by kuairec.train.main

    print_summary(results)


if __name__ == "__main__":
    main()
