"""
Compare metrics across all 4 ablation models.

Looks for JSON files produced by run_all.py (or the individual training
scripts) in LLM-rec/outputs/metrics/ and prints a formatted comparison table.

Supported files (checked in order, first found wins per model):
  A1 Baseline MLP  : a1_baseline_mlp.json  or  metrics_baseline.json
  B1 LLM-MLP       : b1_llm_mlp.json       or  metrics_llm.json
  (A2 / B2 Transformer metrics come from kuairec training health JSON)
"""

import json
from pathlib import Path


def load_history(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _best(history: list[dict]) -> dict:
    return min(history, key=lambda r: r["val_loss"])


def _last(history: list[dict]) -> dict:
    return history[-1]


def summarize(name: str, history: list[dict]) -> dict:
    best = _best(history)
    last = _last(history)
    print(f"  Best  val_loss : {best['val_loss']:.6f}  (epoch {best['epoch']})")
    print(f"  Final val_loss : {last['val_loss']:.6f}  (epoch {last['epoch']})")
    print(f"  Final train_loss: {last['train_loss']:.6f}")
    return {"name": name, "best_val": best["val_loss"], "final_val": last["val_loss"]}


def _load_transformer_health(ckpt_dir: Path) -> dict | None:
    health_path = ckpt_dir / "training_health.json"
    if health_path.exists():
        with open(health_path) as f:
            return json.load(f)
    return None


def main():
    root = Path(__file__).resolve().parents[1]   # .../LLM-rec
    out_dir = root / "outputs" / "metrics"
    ckpt_root = root / "outputs" / "checkpoints"
    kuairec_ckpt_root = root.parent / "kuairec" / "ckpt"

    # -----------------------------------------------------------------------
    # MLP models
    # -----------------------------------------------------------------------
    model_files = [
        ("A1 Baseline MLP",  ["a1_baseline_mlp.json", "metrics_baseline.json"]),
        ("B1 LLM-MLP",       ["b1_llm_mlp.json",      "metrics_llm.json"]),
    ]

    rows: list[dict] = []

    for name, candidates in model_files:
        print(f"\n=== {name} ===")
        history = None
        for fname in candidates:
            history = load_history(out_dir / fname)
            if history is not None:
                break
        if history is None:
            print(f"  [not found — run run_all.py first]")
            rows.append({"name": name, "best_val": None, "final_val": None})
        else:
            rows.append(summarize(name, history))

    # -----------------------------------------------------------------------
    # Transformer models (read from kuairec training_health.json)
    # -----------------------------------------------------------------------
    transformer_models = [
        ("A2 ID-Transformer",  out_dir.parent / "checkpoints" / "a2_id_transformer",  kuairec_ckpt_root),
        ("B2 LLM-Transformer", out_dir.parent / "checkpoints" / "b2_llm_transformer", None),
    ]

    for name, ckpt_dir, fallback_dir in transformer_models:
        print(f"\n=== {name} ===")
        health = _load_transformer_health(ckpt_dir)
        if health is None and fallback_dir is not None:
            health = _load_transformer_health(fallback_dir)
        if health is None:
            print("  [not found — run run_all.py first]")
            rows.append({"name": name, "best_val": None, "final_val": None})
            continue
        best_epoch = health.get("best_epoch", {})
        best_val = best_epoch.get("valid_loss")
        hit10 = best_epoch.get("valid_hit_rate@10", 0.0)
        print(f"  Best valid_loss : {best_val:.6f}" if best_val is not None else "  Best valid_loss: N/A")
        print(f"  Best hit@10     : {hit10:.4f}")
        rows.append({"name": name, "best_val": best_val, "final_val": None, "hit10": hit10})

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 62)
    print(f"{'Model':<28} {'Best val_loss':>14}  {'Final val_loss':>14}")
    print("-" * 62)
    for row in rows:
        bv = f"{row['best_val']:.6f}" if row["best_val"] is not None else "N/A"
        fv = f"{row['final_val']:.6f}" if row.get("final_val") is not None else "—"
        print(f"{row['name']:<28} {bv:>14}  {fv:>14}")
    print("=" * 62)

    # -----------------------------------------------------------------------
    # Pairwise improvement summary (A1 vs B1)
    # -----------------------------------------------------------------------
    a1 = next((r for r in rows if r["name"].startswith("A1")), None)
    b1 = next((r for r in rows if r["name"].startswith("B1")), None)
    if a1 and b1 and a1["best_val"] is not None and b1["best_val"] is not None:
        diff = b1["best_val"] - a1["best_val"]
        print(f"\nB1 LLM-MLP vs A1 Baseline: {diff:+.6f}  ({'worse' if diff > 0 else 'better'} than baseline)")


if __name__ == "__main__":
    main()
