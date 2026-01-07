import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    root = Path(__file__).resolve().parents[1]
    metrics_dir = root / "outputs" / "metrics"
    plots_dir = root / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_hist = load_history(metrics_dir / "metrics_baseline.json")
    llm_hist = load_history(metrics_dir / "metrics_llm.json")

    base_epochs = [r["epoch"] for r in base_hist]
    base_train = [r["train_loss"] for r in base_hist]
    base_val = [r["val_loss"] for r in base_hist]

    llm_epochs = [r["epoch"] for r in llm_hist]
    llm_train = [r["train_loss"] for r in llm_hist]
    llm_val = [r["val_loss"] for r in llm_hist]

    # Plot validation loss
    plt.figure()
    plt.plot(base_epochs, base_val, label="Baseline val_loss")
    plt.plot(llm_epochs, llm_val, label="LLM val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (watch_ratio)")
    plt.title("Validation loss: Baseline vs LLM-enhanced")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "val_loss_comparison.png", dpi=200)
    plt.close()

    # Plot training loss (optional)
    plt.figure()
    plt.plot(base_epochs, base_train, label="Baseline train_loss")
    plt.plot(llm_epochs, llm_train, label="LLM train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (watch_ratio)")
    plt.title("Training loss: Baseline vs LLM-enhanced")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "train_loss_comparison.png", dpi=200)
    plt.close()

    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
