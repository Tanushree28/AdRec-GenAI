import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import KuaiRecTrainDataset, load_kuairec_data
from .model import KuaiRecModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KuaiRec baseline model.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", type=int, default=101)
    parser.add_argument("--hidden_units", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _init_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    log_dir = Path(os.environ.get("TRAIN_LOG_PATH", "./logs"))
    events_dir = Path(os.environ.get("TRAIN_TF_EVENTS_PATH", "./events"))
    ckpt_dir = Path(os.environ.get("TRAIN_CKPT_PATH", "./ckpt"))
    for directory in [log_dir, events_dir, ckpt_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / "train.log"
    log_file = log_file_path.open("w", encoding="utf-8")
    writer = SummaryWriter(str(events_dir))

    args = get_args()
    _init_seed(args.seed)

    data_root = Path(os.environ.get("TRAIN_DATA_PATH", "./kuairec/data"))
    data = load_kuairec_data(data_root)
    dataset = KuaiRecTrainDataset(data, maxlen=args.maxlen)

    if len(dataset) < 2:
        raise ValueError(
            "Not enough training users with at least two interactions. Check the dataset setup."
        )

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=KuaiRecTrainDataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=KuaiRecTrainDataset.collate_fn,
    )

    device = torch.device(args.device)
    model = KuaiRecModel(
        num_items=data.num_items,
        hidden_units=args.hidden_units,
        maxlen=args.maxlen,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
        norm_first=args.norm_first,
    ).to(device)

    if args.state_dict_path:
        state_dict = torch.load(args.state_dict_path, map_location=device)
        model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    print("Start training")
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if args.inference_only:
            break

        train_loss_sum = 0.0
        train_total = 0
        train_correct = 0
        train_hit5 = 0
        train_hit10 = 0

        for batch in tqdm(train_loader, total=len(train_loader)):
            seq = batch["seq"].to(device)
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            seq_repr, pos_embs, neg_embs = model(seq, pos, neg)
            loss, metrics = model.compute_infonce_loss(
                seq_repr, pos_embs, neg_embs, mask, return_metrics=True
            )
            base_loss = loss.item()

            if args.l2_emb > 0:
                reg = torch.tensor(0.0, device=device)
                for param in model.item_embedding.parameters():
                    reg = reg + torch.norm(param, p=2)
                loss = loss + args.l2_emb * reg

            loss.backward()
            optimizer.step()

            batch_total = metrics["total"] if metrics else 0
            batch_acc = (metrics["correct"] / batch_total) if batch_total else 0.0
            batch_hit5 = (metrics["hit_at_5"] / batch_total) if batch_total else 0.0
            batch_hit10 = (metrics["hit_at_10"] / batch_total) if batch_total else 0.0

            train_loss_sum += base_loss
            train_total += batch_total
            train_correct += metrics.get("correct", 0)
            train_hit5 += metrics.get("hit_at_5", 0)
            train_hit10 += metrics.get("hit_at_10", 0)

            log_json = json.dumps(
                {
                    "global_step": global_step,
                    "loss": base_loss,
                    "epoch": epoch,
                    "time": time.time(),
                    "top1_acc": batch_acc,
                    "hit_rate@5": batch_hit5,
                    "hit_rate@10": batch_hit10,
                }
            )
            log_file.write(log_json + "\n")
            log_file.flush()
            print(log_json)

            writer.add_scalar("Loss/train", base_loss, global_step)
            writer.add_scalar("Metrics/train_top1", batch_acc, global_step)
            writer.add_scalar("Metrics/train_hit5", batch_hit5, global_step)
            writer.add_scalar("Metrics/train_hit10", batch_hit10, global_step)

            global_step += 1

        model.eval()
        valid_loss_sum = 0.0
        valid_total = 0
        valid_correct = 0
        valid_hit5 = 0
        valid_hit10 = 0

        for batch in tqdm(valid_loader, total=len(valid_loader)):
            seq = batch["seq"].to(device)
            pos = batch["pos"].to(device)
            neg = batch["neg"].to(device)
            mask = batch["mask"].to(device)

            with torch.no_grad():
                seq_repr, pos_embs, neg_embs = model(seq, pos, neg)
                loss, metrics = model.compute_infonce_loss(
                    seq_repr, pos_embs, neg_embs, mask, return_metrics=True
                )

            valid_loss_sum += loss.item()
            batch_total = metrics["total"] if metrics else 0
            if batch_total:
                valid_total += batch_total
                valid_correct += metrics.get("correct", 0)
                valid_hit5 += metrics.get("hit_at_5", 0)
                valid_hit10 += metrics.get("hit_at_10", 0)

        train_loss_avg = train_loss_sum / max(1, len(train_loader))
        train_top1 = (train_correct / train_total) if train_total else 0.0
        train_hr5 = (train_hit5 / train_total) if train_total else 0.0
        train_hr10 = (train_hit10 / train_total) if train_total else 0.0

        valid_loss_avg = valid_loss_sum / max(1, len(valid_loader))
        valid_top1 = (valid_correct / valid_total) if valid_total else 0.0
        valid_hr5 = (valid_hit5 / valid_total) if valid_total else 0.0
        valid_hr10 = (valid_hit10 / valid_total) if valid_total else 0.0

        writer.add_scalar("Loss/train_epoch", train_loss_avg, epoch)
        writer.add_scalar("Loss/valid", valid_loss_avg, epoch)
        writer.add_scalar("Metrics/train_top1_epoch", train_top1, epoch)
        writer.add_scalar("Metrics/train_hit5_epoch", train_hr5, epoch)
        writer.add_scalar("Metrics/train_hit10_epoch", train_hr10, epoch)
        writer.add_scalar("Metrics/valid_top1_epoch", valid_top1, epoch)
        writer.add_scalar("Metrics/valid_hit5_epoch", valid_hr5, epoch)
        writer.add_scalar("Metrics/valid_hit10_epoch", valid_hr10, epoch)

        summary = {
            "epoch": epoch,
            "phase": "epoch_summary",
            "train_loss": train_loss_avg,
            "train_top1_acc": train_top1,
            "train_hit_rate@5": train_hr5,
            "train_hit_rate@10": train_hr10,
            "valid_loss": valid_loss_avg,
            "valid_top1_acc": valid_top1,
            "valid_hit_rate@5": valid_hr5,
            "valid_hit_rate@10": valid_hr10,
            "time": time.time(),
        }
        log_file.write(json.dumps(summary) + "\n")
        log_file.flush()
        print(
            f"Epoch {epoch}: train_loss={train_loss_avg:.4f}, valid_loss={valid_loss_avg:.4f}, "
            f"train_top1={train_top1:.4f}, valid_top1={valid_top1:.4f}, "
            f"valid_hit@5={valid_hr5:.4f}, valid_hit@10={valid_hr10:.4f}"
        )

        save_dir = ckpt_dir / f"global_step{global_step}.valid_loss={valid_loss_avg:.4f}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")
        metadata = {
            "epoch": epoch,
            "global_step": global_step,
            "valid_loss": valid_loss_avg,
            "train_loss": train_loss_avg,
            "train_hit_rate@5": train_hr5,
            "train_hit_rate@10": train_hr10,
            "valid_hit_rate@5": valid_hr5,
            "valid_hit_rate@10": valid_hr10,
            "args": vars(args),
            "num_users": data.num_users,
            "num_items": data.num_items,
            "timestamp": time.time(),
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2)

    print("Done")
    writer.close()
    log_file.close()
