# LLM-rec/src/train_baseline.py
import json
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from data_loader import load_big_matrix, make_dataloaders
from models.baseline_mlp import BaselineMLP


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
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


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    n = 0
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


def main(config_path: str = "/Users/tanushreenepal/Desktop/AdRec-GenAI/LLM-rec/config/base.yaml"):
    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    df = load_big_matrix(data_cfg["big_matrix_path"], sep=data_cfg["sep"])
    train_loader, val_loader, stats = make_dataloaders(
        df,
        batch_size=train_cfg["batch_size"],
        val_ratio=train_cfg["val_ratio"],
    )

    # Build model
    model = BaselineMLP(
        n_users=stats["n_users"],
        n_items=stats["n_items"],
        user_embedding_dim=model_cfg["user_embedding_dim"],
        item_embedding_dim=model_cfg["item_embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # Outputs
    out_dir = Path("../LLM-rec/outputs")   # NOTE: use ../ not ..LLM-rec
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics" / "metrics_baseline.json"
    ckpt_path    = out_dir / "checkpoints" / "checkpoints_baseline.pt"

    history = []
    start_total = time.time()

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": epoch_time,
        }
        history.append(record)

        # Human-readable + parsable output
        print(json.dumps(record))  # one JSON line per epoch
    
    total_time = time.time() - start_total
    print(f"Total training time: {total_time:.2f} seconds")

    # Save final model & metrics
    torch.save(model.state_dict(), ckpt_path)
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
