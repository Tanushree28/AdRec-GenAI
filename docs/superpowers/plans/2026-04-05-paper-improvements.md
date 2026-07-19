# Paper Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the IEEE paper by adding user-side LLM embeddings, watch_ratio clipping, early stopping with best-checkpoint saving, an ablation study, and popularity-bias metrics.

**Architecture:** Four model variants are trained and compared: Baseline (ID-only), Item-LLM (item embeddings only), User-LLM (user embeddings only), and Full (both). A shared `_train_loop` utility is extracted to avoid duplicating train/eval logic. All scripts run from `LLM-rec/src/`.

**Tech Stack:** PyTorch, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`), pandas, numpy, matplotlib, PyYAML.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `LLM-rec/src/build_user_llm_embeddings.py` | **Create** | Build per-user text descriptions from interaction history + user_features.csv; encode with SentenceTransformer; save `kuairec/embeddings/user_llm_embeddings.npy` |
| `LLM-rec/src/models/user_item_llm_mlp.py` | **Create** | `FullLLMMLP` — user_emb + user_llm_emb + item_emb + item_llm_emb → MLP |
| `LLM-rec/src/models/user_llm_mlp.py` | **Create** | `UserLLMMLP` — user_emb + user_llm_emb + item_emb → MLP (ablation) |
| `LLM-rec/src/train_utils.py` | **Create** | `train_one_epoch`, `eval_one_epoch`, `train_with_early_stopping` shared by all train scripts |
| `LLM-rec/src/train_full_llm.py` | **Create** | Train `FullLLMMLP`; saves `checkpoints_full_llm_best.pt` and `metrics_full_llm.json` |
| `LLM-rec/src/data_loader.py` | **Edit** | Add `watch_ratio_clip` param to `load_big_matrix`; clip at 5.0 |
| `LLM-rec/src/train_baseline.py` | **Edit** | Use `train_with_early_stopping` from `train_utils`; save `checkpoints_baseline_best.pt` |
| `LLM-rec/src/train_llm.py` | **Edit** | Use `train_with_early_stopping`; save `checkpoints_llm_best.pt` |
| `LLM-rec/src/ablation.py` | **Create** | Train all 4 variants, collect best-epoch metrics, write `outputs/metrics/ablation_results.json` |
| `LLM-rec/src/eval_metrics.py` | **Edit** | Accept a checkpoint tag argument; add `diversity_metrics()` (category entropy, long-tail coverage); include both models + ablation variants |
| `LLM-rec/src/compare_metrics.py` | **Edit** | Add ablation table section; add diversity metrics rows |
| `LLM-rec/src/plot_metrics.py` | **Edit** | Add `plot_ablation_barchart()` and `plot_diversity_barchart()` |
| `LLM-rec/config/base.yaml` | **Edit** | Add `watch_ratio_clip: 5.0`, `early_stopping_patience: 5` |

---

## Task 1: watch_ratio clipping in `data_loader.py`

**Files:**
- Modify: `LLM-rec/src/data_loader.py`
- Modify: `LLM-rec/config/base.yaml`

- [ ] **Step 1: Add `watch_ratio_clip` to config**

Edit `LLM-rec/config/base.yaml` — replace the `data:` block:

```yaml
data:
  big_matrix_path: "/Users/tanushreenepal/Desktop/AdRec-GenAI/kuairec/data/big_matrix.csv"
  small_matrix_path: "/Users/tanushreenepal/Desktop/AdRec-GenAI/kuairec/data/small_matrix.csv"
  sep: ","
  watch_ratio_clip: 5.0   # clip extreme outliers (max raw value is 573)

training:
  batch_size: 4096
  num_epochs: 30
  learning_rate: 0.001
  val_ratio: 0.1
  early_stopping_patience: 5

model:
  user_embedding_dim: 64
  item_embedding_dim: 64
  llm_embedding_dim: 384
  hidden_dim: 128
```

- [ ] **Step 2: Add clip param to `load_big_matrix`**

Replace the `load_big_matrix` function in `LLM-rec/src/data_loader.py`:

```python
def load_big_matrix(path: str, sep: str = ",", watch_ratio_clip: float = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=sep,
        dtype={"user_id": "int32", "video_id": "int32"},
        usecols=["user_id", "video_id", "watch_ratio"],
    )
    if watch_ratio_clip is not None:
        df["watch_ratio"] = df["watch_ratio"].clip(upper=watch_ratio_clip)
    return df
```

- [ ] **Step 3: Verify clipping works**

Run from `LLM-rec/src/`:
```bash
python - <<'EOF'
from data_loader import load_big_matrix
df = load_big_matrix(
    "/Users/tanushreenepal/Desktop/AdRec-GenAI/kuairec/data/big_matrix.csv",
    watch_ratio_clip=5.0
)
print("max watch_ratio:", df["watch_ratio"].max())   # should be 5.0
print("mean watch_ratio:", df["watch_ratio"].mean()) # should be ~0.7
EOF
```
Expected output: `max watch_ratio: 5.0`

---

## Task 2: Shared training utilities in `train_utils.py`

**Files:**
- Create: `LLM-rec/src/train_utils.py`

- [ ] **Step 1: Create `train_utils.py`**

```python
# LLM-rec/src/train_utils.py
"""Shared train/eval loop used by all train_*.py scripts."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
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


def eval_one_epoch(model: nn.Module, loader, criterion, device) -> float:
    model.eval()
    total_loss, n = 0.0, 0
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


def train_with_early_stopping(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs: int,
    patience: int,
    best_ckpt_path: Path,
    final_ckpt_path: Path,
    metrics_path: Path,
) -> list[dict]:
    """
    Train for up to num_epochs; stop early if val_loss doesn't improve for `patience` epochs.
    Saves best-val-epoch weights to best_ckpt_path and final weights to final_ckpt_path.
    Returns history list of per-epoch dicts.
    """
    history = []
    best_val_loss = float("inf")
    no_improve = 0
    start_total = time.time()

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_one_epoch(model, val_loader,   criterion, device)
        epoch_time = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time,
        }
        history.append(record)
        print(json.dumps(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"Total training time: {time.time() - start_total:.2f}s")
    torch.save(model.state_dict(), final_ckpt_path)
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)
    return history
```

- [ ] **Step 2: Verify import**

```bash
python -c "from train_utils import train_one_epoch, eval_one_epoch, train_with_early_stopping; print('OK')"
```
Expected: `OK`

---

## Task 3: Update `train_baseline.py` and `train_llm.py` to use early stopping

**Files:**
- Modify: `LLM-rec/src/train_baseline.py`
- Modify: `LLM-rec/src/train_llm.py`

- [ ] **Step 1: Rewrite `train_baseline.py`**

Replace the entire file:

```python
# LLM-rec/src/train_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path

from data_loader import load_big_matrix, make_dataloaders
from models.baseline_mlp import BaselineMLP
from train_utils import train_with_early_stopping


def main(config_path: str = None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_big_matrix(
        data_cfg["big_matrix_path"],
        sep=data_cfg["sep"],
        watch_ratio_clip=data_cfg.get("watch_ratio_clip"),
    )
    train_loader, val_loader, stats = make_dataloaders(
        df, batch_size=train_cfg["batch_size"], val_ratio=train_cfg["val_ratio"]
    )

    model = BaselineMLP(
        n_users=stats["n_users"],
        n_items=stats["n_items"],
        user_embedding_dim=model_cfg["user_embedding_dim"],
        item_embedding_dim=model_cfg["item_embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=train_cfg["num_epochs"],
        patience=train_cfg.get("early_stopping_patience", 5),
        best_ckpt_path=out_dir / "checkpoints" / "checkpoints_baseline_best.pt",
        final_ckpt_path=out_dir / "checkpoints" / "checkpoints_baseline.pt",
        metrics_path=out_dir / "metrics" / "metrics_baseline.json",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rewrite `train_llm.py`**

Replace the entire file:

```python
# LLM-rec/src/train_llm.py
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path

from data_loader import load_big_matrix, make_dataloaders
from llm_features import load_item_llm_embeddings
from models.llm_mlp import LLMEnhancedMLP
from train_utils import train_with_early_stopping


def main(config_path: str = None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_big_matrix(
        data_cfg["big_matrix_path"],
        sep=data_cfg["sep"],
        watch_ratio_clip=data_cfg.get("watch_ratio_clip"),
    )
    train_loader, val_loader, stats = make_dataloaders(
        df, batch_size=train_cfg["batch_size"], val_ratio=train_cfg["val_ratio"]
    )

    src_dir = Path(__file__).resolve().parent
    llm_emb_path = src_dir.parents[1] / "kuairec" / "embeddings" / "item_llm_embeddings.npy"
    item_llm = load_item_llm_embeddings(str(llm_emb_path))
    n_items = stats["n_items"]
    if item_llm.shape[0] < n_items:
        pad = torch.zeros((n_items - item_llm.shape[0], item_llm.shape[1]))
        item_llm = torch.cat([item_llm, pad], dim=0)
    elif item_llm.shape[0] > n_items:
        item_llm = item_llm[:n_items]

    model = LLMEnhancedMLP(
        n_users=stats["n_users"],
        n_items=n_items,
        user_embedding_dim=model_cfg["user_embedding_dim"],
        item_embedding_dim=model_cfg["item_embedding_dim"],
        llm_embedding_dim=model_cfg["llm_embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        item_llm_embeddings=item_llm,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=train_cfg["num_epochs"],
        patience=train_cfg.get("early_stopping_patience", 5),
        best_ckpt_path=out_dir / "checkpoints" / "checkpoints_llm_best.pt",
        final_ckpt_path=out_dir / "checkpoints" / "checkpoints_llm.pt",
        metrics_path=out_dir / "metrics" / "metrics_llm.json",
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify both scripts parse without error**

```bash
python -c "import train_baseline; print('baseline OK')"
python -c "import train_llm; print('llm OK')"
```
Expected: `baseline OK` then `llm OK`

---

## Task 4: Build user LLM embeddings

**Files:**
- Create: `LLM-rec/src/build_user_llm_embeddings.py`

User text description is built from `user_features.csv` (activity degree, follower count range, register days range) and the top-3 most-watched first-level categories per user from `big_matrix.csv` joined with `kuairec_caption_category.csv`.

- [ ] **Step 1: Create `build_user_llm_embeddings.py`**

```python
# LLM-rec/src/build_user_llm_embeddings.py
"""
Build one text description per user from:
  - user_features.csv  (activity, follower/fan counts, register days)
  - big_matrix.csv + kuairec_caption_category.csv  (top-3 watched categories)

Encodes with the same SentenceTransformer used for items.
Output: kuairec/embeddings/user_llm_embeddings.npy  shape [n_users, 384]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def build_user_text(row: pd.Series, top_cats: list[str]) -> str:
    parts = []

    activity = str(row.get("user_active_degree", "")).strip()
    if activity:
        parts.append(f"activity: {activity}")

    fans_range = str(row.get("fans_user_num_range", "")).strip()
    if fans_range and fans_range != "nan":
        parts.append(f"fans: {fans_range}")

    follow_range = str(row.get("follow_user_num_range", "")).strip()
    if follow_range and follow_range != "nan":
        parts.append(f"following: {follow_range}")

    reg_range = str(row.get("register_days_range", "")).strip()
    if reg_range and reg_range != "nan":
        parts.append(f"account age: {reg_range}")

    if top_cats:
        parts.append("likes: " + ", ".join(top_cats))

    return " | ".join(parts) if parts else "unknown user"


def main():
    root      = Path(__file__).resolve().parents[2]   # AdRec-GenAI/
    data_dir  = root / "kuairec" / "data"
    out_dir   = root / "kuairec" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / "user_llm_embeddings.npy"

    print("Loading data...")
    matrix_df  = pd.read_csv(data_dir / "big_matrix.csv",
                              usecols=["user_id", "video_id", "watch_ratio"],
                              dtype={"user_id": "int32", "video_id": "int32"})
    user_feat  = pd.read_csv(data_dir / "user_features.csv")
    caption_df = pd.read_csv(data_dir / "kuairec_caption_category.csv",
                              sep=None, engine="python", on_bad_lines="skip")
    caption_df.columns = [c.strip() for c in caption_df.columns]

    # Map video_id → first_level_category_name
    vid_col = "video_id" if "video_id" in caption_df.columns else "item_id"
    caption_df[vid_col] = pd.to_numeric(caption_df[vid_col], errors="coerce")
    caption_df = caption_df.dropna(subset=[vid_col])
    caption_df[vid_col] = caption_df[vid_col].astype(int)
    vid2cat = caption_df.set_index(vid_col)["first_level_category_name"].to_dict()

    # Top-3 categories per user (by total watch_ratio)
    matrix_df["category"] = matrix_df["video_id"].map(vid2cat)
    cat_watch = (matrix_df.dropna(subset=["category"])
                           .groupby(["user_id", "category"])["watch_ratio"]
                           .sum())
    user_top_cats: dict[int, list[str]] = {}
    for uid, grp in cat_watch.groupby(level=0):
        user_top_cats[int(uid)] = grp.nlargest(3).index.get_level_values("category").tolist()

    # Build text per user
    user_feat["user_id"] = user_feat["user_id"].astype(int)
    user_feat = user_feat.set_index("user_id")
    max_user_id = int(user_feat.index.max())
    print(f"Building texts for {len(user_feat)} users (max_id={max_user_id})...")

    texts: list[str] = []
    user_ids_ordered: list[int] = sorted(user_feat.index.tolist())
    for uid in tqdm(user_ids_ordered):
        row = user_feat.loc[uid]
        top_cats = user_top_cats.get(uid, [])
        texts.append(build_user_text(row, top_cats))

    # Encode
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading SentenceTransformer: {model_name}")
    model = SentenceTransformer(model_name)

    batch_size = 128
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        embs = model.encode(texts[i:i+batch_size], batch_size=batch_size,
                            show_progress_bar=False, convert_to_numpy=True)
        all_embs.append(embs)
    all_embs = np.vstack(all_embs)   # [n_users_in_features, 384]

    # Build [n_users_total, 384] array indexed by user_id
    n_users = max_user_id + 1
    emb_dim = all_embs.shape[1]
    user_emb_arr = np.zeros((n_users, emb_dim), dtype=np.float32)
    for i, uid in enumerate(user_ids_ordered):
        if 0 <= uid < n_users:
            user_emb_arr[uid] = all_embs[i]

    print(f"Final user_emb shape: {user_emb_arr.shape}")
    np.save(out_path, user_emb_arr)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
cd /Users/tanushreenepal/Desktop/AdRec-GenAI/LLM-rec/src
python build_user_llm_embeddings.py
```

Expected: prints shape `(7177, 384)` (or similar) and saves `kuairec/embeddings/user_llm_embeddings.npy`.

- [ ] **Step 3: Verify output**

```bash
python -c "
import numpy as np
arr = np.load('../../kuairec/embeddings/user_llm_embeddings.npy')
print('shape:', arr.shape)         # expect (N, 384)
print('non-zero rows:', (arr.sum(axis=1) != 0).sum())
"
```

---

## Task 5: New model variants for ablation

**Files:**
- Create: `LLM-rec/src/models/user_llm_mlp.py`
- Create: `LLM-rec/src/models/user_item_llm_mlp.py`

- [ ] **Step 1: Create `user_llm_mlp.py`** (user LLM + item ID — no item LLM)

```python
# LLM-rec/src/models/user_llm_mlp.py
import torch
import torch.nn as nn


class UserLLMMLP(nn.Module):
    """Ablation: user_emb + frozen user_llm_emb + item_emb (no item LLM)."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        llm_embedding_dim: int,
        hidden_dim: int,
        user_llm_embeddings: torch.Tensor,   # shape [n_users, llm_dim]
    ):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, user_embedding_dim)
        self.item_emb  = nn.Embedding(n_items, item_embedding_dim)
        assert user_llm_embeddings.shape[0] == n_users, "n_users mismatch"
        self.register_buffer("user_llm", user_llm_embeddings)

        input_dim = user_embedding_dim + llm_embedding_dim + item_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids):
        u_id  = self.user_emb(user_ids)
        u_llm = self.user_llm[user_ids]
        v     = self.item_emb(item_ids)
        x     = torch.cat([u_id, u_llm, v], dim=-1)
        return self.mlp(x).squeeze(-1)
```

- [ ] **Step 2: Create `user_item_llm_mlp.py`** (full model — user LLM + item LLM)

```python
# LLM-rec/src/models/user_item_llm_mlp.py
import torch
import torch.nn as nn


class FullLLMMLP(nn.Module):
    """Full model: user_emb + user_llm_emb + item_emb + item_llm_emb."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        llm_embedding_dim: int,
        hidden_dim: int,
        user_llm_embeddings: torch.Tensor,   # shape [n_users, llm_dim]
        item_llm_embeddings: torch.Tensor,   # shape [n_items, llm_dim]
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, user_embedding_dim)
        self.item_emb = nn.Embedding(n_items, item_embedding_dim)
        assert user_llm_embeddings.shape[0] == n_users, "n_users mismatch"
        assert item_llm_embeddings.shape[0] == n_items, "n_items mismatch"
        self.register_buffer("user_llm", user_llm_embeddings)
        self.register_buffer("item_llm", item_llm_embeddings)

        input_dim = user_embedding_dim + llm_embedding_dim + item_embedding_dim + llm_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids):
        u_id  = self.user_emb(user_ids)
        u_llm = self.user_llm[user_ids]
        v_id  = self.item_emb(item_ids)
        v_llm = self.item_llm[item_ids]
        x     = torch.cat([u_id, u_llm, v_id, v_llm], dim=-1)
        return self.mlp(x).squeeze(-1)
```

- [ ] **Step 3: Verify imports**

```bash
python -c "
from models.user_llm_mlp import UserLLMMLP
from models.user_item_llm_mlp import FullLLMMLP
print('models OK')
"
```

---

## Task 6: Train full LLM model (`train_full_llm.py`)

**Files:**
- Create: `LLM-rec/src/train_full_llm.py`

- [ ] **Step 1: Create `train_full_llm.py`**

```python
# LLM-rec/src/train_full_llm.py
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path

from data_loader import load_big_matrix, make_dataloaders
from llm_features import load_item_llm_embeddings
from models.user_item_llm_mlp import FullLLMMLP
from train_utils import train_with_early_stopping


def _load_and_align(path: Path, n: int, device) -> torch.Tensor:
    """Load .npy LLM embedding, pad/truncate to exactly n rows."""
    import numpy as np
    arr = torch.tensor(np.load(path), dtype=torch.float32)
    if arr.shape[0] < n:
        pad = torch.zeros((n - arr.shape[0], arr.shape[1]))
        arr = torch.cat([arr, pad], dim=0)
    elif arr.shape[0] > n:
        arr = arr[:n]
    return arr


def main(config_path: str = None):
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "base.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_big_matrix(
        data_cfg["big_matrix_path"],
        sep=data_cfg["sep"],
        watch_ratio_clip=data_cfg.get("watch_ratio_clip"),
    )
    train_loader, val_loader, stats = make_dataloaders(
        df, batch_size=train_cfg["batch_size"], val_ratio=train_cfg["val_ratio"]
    )

    root    = Path(__file__).resolve().parents[2]
    emb_dir = root / "kuairec" / "embeddings"

    item_llm = _load_and_align(emb_dir / "item_llm_embeddings.npy", stats["n_items"], device)
    user_llm = _load_and_align(emb_dir / "user_llm_embeddings.npy", stats["n_users"], device)

    model = FullLLMMLP(
        n_users=stats["n_users"],
        n_items=stats["n_items"],
        user_embedding_dim=model_cfg["user_embedding_dim"],
        item_embedding_dim=model_cfg["item_embedding_dim"],
        llm_embedding_dim=model_cfg["llm_embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        user_llm_embeddings=user_llm,
        item_llm_embeddings=item_llm,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_with_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=train_cfg["num_epochs"],
        patience=train_cfg.get("early_stopping_patience", 5),
        best_ckpt_path=out_dir / "checkpoints" / "checkpoints_full_llm_best.pt",
        final_ckpt_path=out_dir / "checkpoints" / "checkpoints_full_llm.pt",
        metrics_path=out_dir / "metrics" / "metrics_full_llm.json",
    )


if __name__ == "__main__":
    main()
```

---

## Task 7: Ablation script

**Files:**
- Create: `LLM-rec/src/ablation.py`

Trains all 4 variants sequentially with the same data split (seed=42) and writes `outputs/metrics/ablation_results.json`. Each variant uses best-checkpoint from early stopping.

- [ ] **Step 1: Create `ablation.py`**

```python
# LLM-rec/src/ablation.py
"""
Trains all 4 ablation variants with seed=42 val split and early stopping.
Writes outputs/metrics/ablation_results.json with best-epoch val_loss per variant.

Variants:
  1. Baseline      — user_emb + item_emb
  2. Item-LLM      — user_emb + item_emb + item_llm_emb
  3. User-LLM      — user_emb + user_llm_emb + item_emb
  4. Full-LLM      — user_emb + user_llm_emb + item_emb + item_llm_emb
"""
from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from pathlib import Path

from data_loader import load_big_matrix, make_dataloaders
from models.baseline_mlp import BaselineMLP
from models.llm_mlp import LLMEnhancedMLP
from models.user_llm_mlp import UserLLMMLP
from models.user_item_llm_mlp import FullLLMMLP
from train_utils import train_with_early_stopping


def _load_and_align(path: Path, n: int) -> torch.Tensor:
    arr = torch.tensor(np.load(path), dtype=torch.float32)
    if arr.shape[0] < n:
        arr = torch.cat([arr, torch.zeros((n - arr.shape[0], arr.shape[1]))], dim=0)
    return arr[:n]


def main():
    src_dir = Path(__file__).resolve().parent
    llm_root = src_dir.parent
    root = src_dir.parents[1]

    config_path = llm_root / "config" / "base.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    df = load_big_matrix(
        data_cfg["big_matrix_path"],
        sep=data_cfg["sep"],
        watch_ratio_clip=data_cfg.get("watch_ratio_clip"),
    )
    # Fixed seed so all variants see the same val set
    train_loader, val_loader, stats = make_dataloaders(
        df, batch_size=train_cfg["batch_size"], val_ratio=train_cfg["val_ratio"], seed=42
    )

    emb_dir   = root / "kuairec" / "embeddings"
    item_llm  = _load_and_align(emb_dir / "item_llm_embeddings.npy", stats["n_items"])
    user_llm  = _load_and_align(emb_dir / "user_llm_embeddings.npy", stats["n_users"])

    out_dir   = llm_root / "outputs"
    ckpt_dir  = out_dir / "checkpoints"
    met_dir   = out_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    patience    = train_cfg.get("early_stopping_patience", 5)
    num_epochs  = train_cfg["num_epochs"]
    lr          = train_cfg["learning_rate"]
    criterion   = nn.MSELoss()

    u_dim = model_cfg["user_embedding_dim"]
    i_dim = model_cfg["item_embedding_dim"]
    l_dim = model_cfg["llm_embedding_dim"]
    h_dim = model_cfg["hidden_dim"]
    n_u   = stats["n_users"]
    n_i   = stats["n_items"]

    variants = [
        ("Baseline",  "baseline",
         BaselineMLP(n_u, n_i, u_dim, i_dim, h_dim)),
        ("Item-LLM",  "item_llm",
         LLMEnhancedMLP(n_u, n_i, u_dim, i_dim, l_dim, h_dim, item_llm.clone())),
        ("User-LLM",  "user_llm",
         UserLLMMLP(n_u, n_i, u_dim, i_dim, l_dim, h_dim, user_llm.clone())),
        ("Full-LLM",  "full_llm",
         FullLLMMLP(n_u, n_i, u_dim, i_dim, l_dim, h_dim, user_llm.clone(), item_llm.clone())),
    ]

    ablation_summary = []

    for name, tag, model in variants:
        print(f"{'='*50}\nTraining: {name}\n{'='*50}")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        history = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=num_epochs,
            patience=patience,
            best_ckpt_path=ckpt_dir / f"checkpoints_{tag}_best.pt",
            final_ckpt_path=ckpt_dir / f"checkpoints_{tag}.pt",
            metrics_path=met_dir / f"metrics_{tag}.json",
        )
        best = min(history, key=lambda r: r["val_loss"])
        ablation_summary.append({
            "variant": name,
            "tag": tag,
            "best_val_loss": best["val_loss"],
            "best_epoch": best["epoch"],
            "total_epochs": len(history),
        })
        print(f"  Best val_loss: {best['val_loss']:.4f} at epoch {best['epoch']}\n")

    out_path = met_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(ablation_summary, f, indent=2)
    print(f"Ablation results saved → {out_path}")

    print("\n=== Ablation Summary ===")
    for r in ablation_summary:
        print(f"  {r['variant']:<12}  best_val_loss={r['best_val_loss']:.4f}  epoch={r['best_epoch']}/{r['total_epochs']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import (don't run training yet)**

```bash
python -c "import ablation; print('ablation OK')"
```

---

## Task 8: Add diversity metrics to `eval_metrics.py`

**Files:**
- Modify: `LLM-rec/src/eval_metrics.py`

Diversity = how spread are the top-10 recommended items across KuaiRec's `first_level_category_name`. Two metrics:
- **Category Entropy**: Shannon entropy of predicted top-10 category distribution (averaged over users)
- **Long-tail Coverage**: fraction of recommended items that are in the bottom 80% of item popularity

- [ ] **Step 1: Add `diversity_metrics` function and update `main`**

Add the following function to `eval_metrics.py` (after `ranking_metrics`):

```python
def diversity_metrics(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    y_pred: np.ndarray,
    vid2cat: dict,
    item_popularity: dict,
    popularity_threshold: int,
    k: int = 10,
) -> dict:
    """
    Compute per-user top-k category entropy and long-tail coverage.
    - vid2cat: dict mapping video_id (int) → category (str)
    - item_popularity: dict mapping video_id (int) → interaction count
    - popularity_threshold: items with count <= this are "long-tail"
    """
    entropy_list, longtail_list = [], []

    for uid in np.unique(user_ids):
        mask = user_ids == uid
        if mask.sum() < k:
            continue
        top_idx  = np.argsort(y_pred[mask])[::-1][:k]
        top_items = item_ids[mask][top_idx]

        cats = [vid2cat.get(int(iid), "unknown") for iid in top_items]
        from collections import Counter
        cat_counts = Counter(cats)
        total = sum(cat_counts.values())
        probs = np.array([v / total for v in cat_counts.values()])
        entropy = float(-np.sum(probs * np.log2(probs + 1e-9)))
        entropy_list.append(entropy)

        longtail_hits = sum(
            1 for iid in top_items if item_popularity.get(int(iid), 0) <= popularity_threshold
        )
        longtail_list.append(longtail_hits / k)

    return {
        "category_entropy": float(np.mean(entropy_list))   if entropy_list  else 0.0,
        "longtail_coverage": float(np.mean(longtail_list)) if longtail_list else 0.0,
    }
```

Then update `main()` in `eval_metrics.py` to:
1. Load `kuairec_caption_category.csv` and build `vid2cat`
2. Compute `item_popularity` from `big_matrix.csv` (interaction counts per video_id)
3. Set `popularity_threshold` = 20th-percentile count (items seen by fewer users)
4. Call `diversity_metrics(...)` for each model and add `"diversity"` key to result JSON

Replace the `main()` function in `eval_metrics.py`:

```python
def main():
    src_dir  = Path(__file__).resolve().parent
    root     = src_dir.parents[1]
    llm_root = src_dir.parent

    config_path = llm_root / "config" / "base.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import pandas as pd
    # Build vid2cat and item_popularity
    caption_df = pd.read_csv(
        root / "kuairec" / "data" / "kuairec_caption_category.csv",
        sep=None, engine="python", on_bad_lines="skip"
    )
    caption_df.columns = [c.strip() for c in caption_df.columns]
    vid_col = "video_id" if "video_id" in caption_df.columns else "item_id"
    caption_df[vid_col] = pd.to_numeric(caption_df[vid_col], errors="coerce")
    caption_df = caption_df.dropna(subset=[vid_col])
    vid2cat = caption_df.set_index(vid_col.strip())["first_level_category_name"].to_dict()
    vid2cat = {int(k): v for k, v in vid2cat.items()}

    matrix_df = load_big_matrix(data_cfg["big_matrix_path"], sep=data_cfg["sep"])
    pop_counts = matrix_df.groupby("video_id").size().to_dict()
    popularity_threshold = int(np.percentile(list(pop_counts.values()), 20))
    print(f"Long-tail threshold (20th pct): {popularity_threshold} interactions\n")

    df = load_big_matrix(
        data_cfg["big_matrix_path"],
        sep=data_cfg["sep"],
        watch_ratio_clip=data_cfg.get("watch_ratio_clip"),
    )
    _, val_loader, stats = make_dataloaders(
        df, batch_size=train_cfg["batch_size"], val_ratio=train_cfg["val_ratio"], seed=42
    )

    ckpt_dir     = llm_root / "outputs" / "checkpoints"
    out_dir      = llm_root / "outputs" / "metrics"
    llm_emb_path = root / "kuairec" / "embeddings" / "item_llm_embeddings.npy"

    models_to_eval = [
        ("Baseline MLP",     "baseline", build_baseline(stats, model_cfg, ckpt_dir / "checkpoints_baseline_best.pt", device)),
        ("LLM-Enhanced MLP", "llm",      build_llm(stats, model_cfg, ckpt_dir / "checkpoints_llm_best.pt", llm_emb_path, device)),
    ]

    for name, tag, model in models_to_eval:
        print(f"=== Evaluating: {name} ===")
        user_ids, item_ids, y_true, y_pred = collect_predictions(model, val_loader, device)

        reg  = regression_metrics(y_true, y_pred)
        rank = ranking_metrics(user_ids, y_true, y_pred)
        div  = diversity_metrics(user_ids, item_ids, y_pred, vid2cat, pop_counts, popularity_threshold)

        print(f"  MSE={reg['MSE']:.4f}  RMSE={reg['RMSE']:.4f}  MAE={reg['MAE']:.4f}")
        print(f"  CategoryEntropy={div['category_entropy']:.4f}  LongtailCoverage={div['longtail_coverage']:.4f}\n")

        result = {"model": name, "regression": reg, "ranking": rank, "diversity": div}
        out_path = out_dir / f"eval_results_{tag}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {out_path}\n")

    print("Done.")
```

---

## Task 9: Update `compare_metrics.py` with ablation table and diversity rows

**Files:**
- Modify: `LLM-rec/src/compare_metrics.py`

- [ ] **Step 1: Add ablation summary section and diversity rows**

Add these two functions to `compare_metrics.py` (before `main()`):

```python
def print_ablation_table(ablation_results: list):
    print("=" * 55)
    print("ABLATION STUDY  (best-epoch val loss, seed=42 split)")
    print("=" * 55)
    header = f"{'Variant':<14} {'Best Val Loss':>14} {'Best Epoch':>11} {'Total Epochs':>13}"
    print(header)
    print("-" * len(header))
    best_loss = min(r["best_val_loss"] for r in ablation_results)
    for r in ablation_results:
        marker = " ◄ best" if r["best_val_loss"] == best_loss else ""
        print(
            f"{r['variant']:<14} {r['best_val_loss']:>14.4f} "
            f"{r['best_epoch']:>11} {r['total_epochs']:>13}{marker}"
        )
    print()


def print_diversity_rows(base: dict, llm: dict):
    print("=" * 55)
    print("DIVERSITY METRICS (higher = more diverse recommendations)")
    print("=" * 55)
    for metric_key, label in [
        ("category_entropy",  "Category Entropy"),
        ("longtail_coverage", "Longtail Coverage"),
    ]:
        b = base.get("diversity", {}).get(metric_key, 0.0)
        l = llm.get("diversity", {}).get(metric_key, 0.0)
        pct = (l - b) / (b + 1e-9) * 100
        arrow = "↑ better" if pct > 0 else "↓ worse"
        print(f"  {label:<22} Baseline={b:.4f}  LLM={l:.4f}  {pct:+.2f}%  {arrow}")
    print()
```

And add to `main()` in `compare_metrics.py` (after the existing full metric comparison block):

```python
    # ── Ablation table ────────────────────────────────────────────────────────
    ablation_path = out_dir / "metrics" / "ablation_results.json"
    if ablation_path.exists():
        print_ablation_table(load_json(ablation_path))
    else:
        print("  (run ablation.py to see ablation results)\n")

    # ── Diversity rows ────────────────────────────────────────────────────────
    if eval_base_path.exists() and eval_llm_path.exists():
        print_diversity_rows(base, llm)
```

---

## Task 10: Update `plot_metrics.py` with ablation and diversity bar charts

**Files:**
- Modify: `LLM-rec/src/plot_metrics.py`

- [ ] **Step 1: Add `plot_ablation_barchart` function**

Add after the existing `plot_metric_barchart` function:

```python
def plot_ablation_barchart(ablation_results: list, plots_dir: Path):
    """Horizontal bar chart of best val_loss per ablation variant (lower = better)."""
    names  = [r["variant"]  for r in ablation_results]
    losses = [r["best_val_loss"] for r in ablation_results]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(names)]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, losses, color=colors)
    ax.set_xlabel("Best Validation MSE (lower is better)")
    ax.set_title("Ablation Study — Best Epoch Val Loss per Variant")
    ax.invert_yaxis()
    for bar, val in zip(bars, losses):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    out = plots_dir / "ablation_barchart.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved ablation bar chart → {out}")


def plot_diversity_barchart(eval_base: dict, eval_llm: dict, plots_dir: Path):
    """Grouped bar chart for diversity metrics."""
    metrics = ["category_entropy", "longtail_coverage"]
    labels  = ["Category Entropy", "Longtail Coverage"]
    base_vals = [eval_base.get("diversity", {}).get(m, 0.0) for m in metrics]
    llm_vals  = [eval_llm.get("diversity", {}).get(m, 0.0)  for m in metrics]

    x     = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars_b = ax.bar(x - width / 2, base_vals, width, label="Baseline MLP",     color="#4C72B0")
    bars_l = ax.bar(x + width / 2, llm_vals,  width, label="LLM-Enhanced MLP", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("Diversity Metrics (higher is better)")
    ax.legend()
    for bar in list(bars_b) + list(bars_l):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out = plots_dir / "diversity_barchart.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"  Saved diversity bar chart → {out}")
```

- [ ] **Step 2: Call both new functions from `main()` in `plot_metrics.py`**

Append to the end of `main()` (after the existing bar chart block):

```python
    # Ablation bar chart
    ablation_path = metrics_dir / "ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_data = json.load(f)
        plot_ablation_barchart(ablation_data, plots_dir)
    else:
        print("  Skipping ablation chart — run ablation.py first")

    # Diversity bar chart (reuses eval_base and eval_llm loaded earlier)
    if eval_base_path.exists() and eval_llm_path.exists():
        plot_diversity_barchart(eval_base, eval_llm, plots_dir)
    else:
        print("  Skipping diversity chart — run eval_metrics.py first")
```

---

## Task 11: Retrain all models and regenerate all outputs

This is the integration step — run everything in order.

- [ ] **Step 1: Retrain baseline (with early stopping + clipping)**

```bash
cd /Users/tanushreenepal/Desktop/AdRec-GenAI/LLM-rec/src
python train_baseline.py
```
Expected: stops before epoch 30 when val_loss plateaus; saves `checkpoints_baseline_best.pt`

- [ ] **Step 2: Retrain item-LLM model**

```bash
python train_llm.py
```
Expected: stops early; saves `checkpoints_llm_best.pt`

- [ ] **Step 3: Run ablation (trains all 4 variants)**

```bash
python ablation.py
```
Expected: trains Baseline, Item-LLM, User-LLM, Full-LLM sequentially; saves `ablation_results.json`

- [ ] **Step 4: Evaluate and generate all metrics**

```bash
python eval_metrics.py
```
Expected: writes `eval_results_baseline.json` and `eval_results_llm.json` including diversity keys

- [ ] **Step 5: Print paper-ready comparison**

```bash
python compare_metrics.py
```
Expected: prints loss curve summary, full metric table, ablation table, diversity rows, and LaTeX block

- [ ] **Step 6: Regenerate all plots**

```bash
python plot_metrics.py
```
Expected: regenerates `val_loss_comparison.png`, `train_loss_comparison.png`, `metrics_barchart.png`, `ablation_barchart.png`, `diversity_barchart.png`

---

## Verification Checklist

- [ ] `kuairec/embeddings/user_llm_embeddings.npy` exists, shape `(N, 384)` with N >= 7176
- [ ] `outputs/checkpoints/checkpoints_baseline_best.pt` exists
- [ ] `outputs/checkpoints/checkpoints_llm_best.pt` exists
- [ ] `outputs/checkpoints/checkpoints_full_llm_best.pt` exists
- [ ] `outputs/metrics/ablation_results.json` has 4 variants; Full-LLM has lowest best_val_loss
- [ ] `outputs/metrics/eval_results_baseline.json` has `regression`, `ranking`, and `diversity` keys
- [ ] `outputs/metrics/eval_results_llm.json` has `regression`, `ranking`, and `diversity` keys
- [ ] `compare_metrics.py` prints LLM-Enhanced better on MSE/RMSE and diversity metrics
- [ ] `outputs/plots/ablation_barchart.png` exists
- [ ] `outputs/plots/diversity_barchart.png` exists
- [ ] All scripts run without error from `LLM-rec/src/`
