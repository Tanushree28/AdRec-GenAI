"""
Generate all paper figures from actual training results.
Outputs PNG files to LLM-rec/outputs/plots/paper_figures/

Run from project root:
    python LLM-rec/src/generate_paper_figures.py
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "LLM-rec" / "outputs" / "plots" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (colour-blind friendly) ─────────────────────────────────
C_A1    = "#4878CF"   # blue   — A1 Baseline MLP
C_B1    = "#D65F5F"   # red    — B1 LLM-MLP raw
C_B1G   = "#6ACC65"   # green  — B1-Gen generative
C_A2    = "#4878CF"   # blue   — A2 ID-Transformer
C_B2    = "#D65F5F"   # red    — B2 LLM-Transformer raw
C_B2G   = "#6ACC65"   # green  — B2-Gen generative

FONT = {"family": "serif", "size": 11}
matplotlib.rc("font", **FONT)

# ── Hard-coded results (from actual training run) ───────────────────────────
MLP = {
    "A1 Baseline\nMLP":         {"mse": 2.5431, "rmse": 1.5947},
    "B1 LLM-MLP\n(raw)":        {"mse": 2.6879, "rmse": 1.6395},
    "B1-Gen LLM-MLP\n(Mistral-7B)": {"mse": 2.4697, "rmse": 1.5715},
}
TF = {
    "A2 ID-\nTransformer":          {"infonce": 8.8093, "hit10": 0.0025},
    "B2 LLM-Trans.\n(raw)":         {"infonce": 8.5208, "hit10": 0.0080},
    "B2-Gen LLM-Trans.\n(Mistral-7B)": {"infonce": 8.5208, "hit10": 0.0080},
}
DIVERSITY = {
    "A2":    {"coverage": 0.2116, "pop_bias": 0.1882, "long_tail": 0.0121},
    "B2":    {"coverage": 0.2116, "pop_bias": 0.1882, "long_tail": 0.0121},
    "B2-Gen":{"coverage": 0.2116, "pop_bias": 0.1882, "long_tail": 0.0121},
}


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — MLP Regression: MSE and RMSE grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════
def fig2_mlp_regression():
    labels = list(MLP.keys())
    mse_vals  = [MLP[k]["mse"]  for k in labels]
    rmse_vals = [MLP[k]["rmse"] for k in labels]
    colors = [C_A1, C_B1, C_B1G]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(
        "Baseline vs LLM-Enhanced MLP — KuaiRec watch_ratio prediction",
        fontsize=12, fontweight="bold"
    )

    for ax, vals, metric, ylabel in [
        (axes[0], mse_vals,  "MSE (lower is better)",  "MSE"),
        (axes[1], rmse_vals, "RMSE (lower is better)", "RMSE"),
    ]:
        bars = ax.bar(x, vals, width=0.5, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(metric, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(min(vals) * 0.97, max(vals) * 1.04)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        # value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9
            )

    # legend
    patches = [
        mpatches.Patch(color=C_A1,  label="A1 — Baseline MLP (ID only)"),
        mpatches.Patch(color=C_B1,  label="B1 — LLM-MLP (raw profiles)"),
        mpatches.Patch(color=C_B1G, label="B1-Gen — LLM-MLP (Mistral-7B)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    path = OUT_DIR / "fig2_mlp_regression.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Ablation: two side-by-side charts (MLP MSE + Transformer Hit@10)
# ═══════════════════════════════════════════════════════════════════════════
def fig3_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Ablation Study — MLP Regression (left) and Transformer Ranking (right)",
        fontsize=12, fontweight="bold"
    )

    # Left: MLP MSE
    ax = axes[0]
    mlp_labels = list(MLP.keys())
    mlp_mse    = [MLP[k]["mse"] for k in mlp_labels]
    colors_mlp = [C_A1, C_B1, C_B1G]
    bars = ax.barh(mlp_labels, mlp_mse, color=colors_mlp,
                   edgecolor="white", linewidth=0.8)
    ax.set_title("MLP — Best Val MSE (lower is better)", fontsize=10)
    ax.set_xlabel("Validation MSE")
    ax.set_xlim(2.40, 2.75)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, mlp_mse):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    # annotation arrow for B1-Gen
    ax.annotate("Best ✓", xy=(2.4697, 2), xytext=(2.42, 1.7),
                fontsize=8, color="green",
                arrowprops=dict(arrowstyle="->", color="green", lw=1.2))

    # Right: Transformer Hit@10
    ax = axes[1]
    tf_labels = list(TF.keys())
    tf_hit    = [TF[k]["hit10"] for k in tf_labels]
    colors_tf = [C_A2, C_B2, C_B2G]
    bars = ax.bar(np.arange(len(tf_labels)), tf_hit,
                  color=colors_tf, edgecolor="white", linewidth=0.8, width=0.5)
    ax.set_title("Transformer — Hit@10 (higher is better)", fontsize=10)
    ax.set_ylabel("Hit@10")
    ax.set_xticks(np.arange(len(tf_labels)))
    ax.set_xticklabels(tf_labels, fontsize=9)
    ax.set_ylim(0, 0.011)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, tf_hit):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    # +220% annotation
    ax.annotate("+220% vs A2", xy=(1, 0.0080), xytext=(1.3, 0.009),
                fontsize=8, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig3_ablation.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Diversity: coverage, popularity bias, long-tail strength
# ═══════════════════════════════════════════════════════════════════════════
def fig4_diversity():
    div_labels = ["A2\nID-Transformer", "B2\nLLM-Trans. (raw)", "B2-Gen\nLLM-Trans. (gen.)"]
    keys = ["A2", "B2", "B2-Gen"]
    coverage  = [DIVERSITY[k]["coverage"]  for k in keys]
    pop_bias  = [DIVERSITY[k]["pop_bias"]  for k in keys]
    long_tail = [DIVERSITY[k]["long_tail"] for k in keys]
    colors_div = [C_A2, C_B2, C_B2G]

    x = np.arange(len(div_labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(
        "Diversity Metrics — Transformer Models (higher is better for all)",
        fontsize=12, fontweight="bold"
    )

    for ax, vals, title, ylabel in [
        (axes[0], coverage,  "Item Coverage\n(fraction of catalogue)", "Coverage"),
        (axes[1], pop_bias,  "Popularity Bias Score\n(higher = less biased)", "Score"),
        (axes[2], long_tail, "Long-tail Strength\n(fraction of tail items)", "Score"),
    ]:
        bars = ax.bar(x, vals, width=0.5, color=colors_div,
                      edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(div_labels, fontsize=8)
        span = max(vals) - min(vals)
        ax.set_ylim(
            max(0, min(vals) - span * 2),
            max(vals) + span * 2 + 0.001
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + span * 0.1,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # note about identical values
    fig.text(0.5, 0.01,
             "All three Transformer models show identical diversity — LLM augmentation "
             "does not improve or harm diversity at 5 epochs.",
             ha="center", fontsize=9, style="italic", color="gray")

    patches = [
        mpatches.Patch(color=C_A2,  label="A2 — ID-Transformer"),
        mpatches.Patch(color=C_B2,  label="B2 — LLM-Transformer (raw)"),
        mpatches.Patch(color=C_B2G, label="B2-Gen — LLM-Transformer (Mistral-7B)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.06))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    path = OUT_DIR / "fig4_diversity.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# BONUS — Transformer ranking InfoNCE loss curve across epochs
# ═══════════════════════════════════════════════════════════════════════════
def fig_loss_curves():
    """Plot InfoNCE val loss per epoch for A2, B2, B2-Gen."""
    ckpt_root = ROOT / "LLM-rec" / "outputs" / "checkpoints"
    model_map = {
        "A2 ID-Transformer":           ckpt_root / "a2_id_transformer",
        "B2 LLM-Transformer (raw)":    ckpt_root / "b2_llm_transformer",
        "B2-Gen LLM-Transformer (gen)":ckpt_root / "b2_gen_llm_transformer",
    }
    colors_lc = [C_A2, C_B2, C_B2G]
    markers   = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("Transformer — Validation InfoNCE Loss per Epoch", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation InfoNCE Loss (lower is better)")

    for (name, ckpt), color, marker in zip(model_map.items(), colors_lc, markers):
        # Read loss from checkpoint filenames like "global_step20.valid_loss=9.0180"
        losses = {}
        for f in ckpt.glob("global_step*.valid_loss=*"):
            step = int(f.name.split("global_step")[1].split(".")[0])
            loss = float(f.name.split("valid_loss=")[1])
            losses[step] = loss
        if not losses:
            continue
        steps = sorted(losses.keys())
        vals  = [losses[s] for s in steps]
        # steps per epoch ≈ 20 (20 batches × 5 epochs = 100 steps)
        epochs = [s / 20 for s in steps]
        ax.plot(epochs, vals, color=color, marker=marker,
                linewidth=2, markersize=6, label=name)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "fig_loss_curves.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Saving figures to: {OUT_DIR}\n")
    fig2_mlp_regression()
    fig3_ablation()
    fig4_diversity()
    fig_loss_curves()
    print("\nDone. All figures saved.")
