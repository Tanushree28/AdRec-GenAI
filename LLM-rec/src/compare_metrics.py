import json
from pathlib import Path


def load_history(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def summarize(name: str, history):
    best = min(history, key=lambda r: r["val_loss"])
    last = history[-1]
    print(f"=== {name} ===")
    print(f"  Best  val_loss: {best['val_loss']:.4f} (epoch {best['epoch']})")
    print(f"  Final val_loss: {last['val_loss']:.4f} (epoch {last['epoch']})")
    print(f"  Final train_loss: {last['train_loss']:.4f}")
    print()


def main():
    root = Path(__file__).resolve().parents[1]  # .../LLM-rec
    out_dir = root / "outputs" / "metrics"

    baseline_path = out_dir / "metrics_baseline.json"
    llm_path = out_dir / "metrics_llm.json"

    base_hist = load_history(baseline_path)
    llm_hist = load_history(llm_path)

    summarize("Baseline (ID-only)", base_hist)
    summarize("LLM-enhanced (ID + LLM item)", llm_hist)

    # Tiny difference summary
    base_last = base_hist[-1]["val_loss"]
    llm_last = llm_hist[-1]["val_loss"]
    diff = llm_last - base_last
    print("=== Comparison (Final val_loss) ===")
    print(f"  Baseline:   {base_last:.4f}")
    print(f"  LLM model:  {llm_last:.4f}")
    print(f"  LLM - Base: {diff:+.4f} (positive = worse than baseline)")


if __name__ == "__main__":
    main()
