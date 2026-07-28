import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import (
    KuaiRecTrainDataset,
    compute_dataset_statistics,
    downsample_head_items,
    load_kuairec_data,
    trim_user_sequences,
)
from .model import KuaiRecModel
from .model_llm import LLMKuaiRecModel


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _load_eval_metrics() -> Optional[dict]:
    """Best-effort loading of the latest evaluation metrics JSON if it exists."""

    candidates = [
        os.environ.get("EVAL_REC_RESULT_PATH"),
        os.environ.get("EVAL_RESULT_PATH"),
        os.environ.get("EVAL_RESULTS_PATH"),
        str(Path("kuairec/eval_results")),
    ]
    visited: set[Path] = set()
    for candidate in candidates:
        if not candidate:
            continue
        base = Path(candidate).expanduser().resolve()
        potential_files = []
        if base.is_dir():
            potential_files.append(base / "metrics.json")
        else:
            potential_files.append(base)
        for metrics_path in potential_files:
            if metrics_path in visited or not metrics_path.exists():
                continue
            visited.add(metrics_path)
            try:
                with metrics_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                return data
            except Exception:
                continue
    return None


def _detect_training_issues(
    dataset_stats: dict,
    best_epoch: dict,
    epoch_summaries: List[dict],
    eval_metrics: Optional[dict],
    args: argparse.Namespace,
) -> Dict[str, bool]:
    issues: Dict[str, bool] = {}
    coverage = None
    ndcg = None
    if eval_metrics:
        coverage = eval_metrics.get("catalog_coverage")
        ndcg = eval_metrics.get("ndcg@k") or eval_metrics.get("ndcg")

    first_loss = epoch_summaries[0]["valid_loss"] if epoch_summaries else best_epoch["valid_loss"]
    best_loss = best_epoch["valid_loss"]
    last_epoch = epoch_summaries[-1] if epoch_summaries else best_epoch
    improvement = (first_loss - best_loss) / max(first_loss, 1e-8)
    hit_rate = best_epoch.get("valid_hit_rate@10", 0.0)
    pop_bias = dataset_stats.get("popularity_bias_score", 0.0)
    long_tail_ratio = dataset_stats.get("long_tail_ratio", 0.0)
    avg_len = dataset_stats.get("avg_sequence_length", 0.0)
    recent_ratio = (
        dataset_stats.get("personalization_metrics", {}).get("recent_30d_interaction_ratio")
    )

    issues["popularity_collapse"] = bool(
        (coverage is not None and coverage < 0.35)
        or (pop_bias > 0.6 and hit_rate < 0.02)
    )
    issues["underfitting"] = bool(improvement < 0.05 or best_loss > 5 or hit_rate < 0.02)
    issues["overfitting"] = bool(
        len(epoch_summaries) >= 2
        and last_epoch["valid_loss"] > best_loss * 1.05
        and last_epoch.get("train_hit_rate@10", 0.0)
        > last_epoch.get("valid_hit_rate@10", 0.0) * 1.5
    )
    issues["insufficient_negative_sampling"] = bool(hit_rate < 0.02 and long_tail_ratio > 0.3)
    issues["too_long_sequences"] = bool(avg_len > args.maxlen * 1.5)
    issues["poor_diversity"] = bool(
        (coverage is not None and coverage < 0.4) or pop_bias > 0.65
    )
    issues["low_personalization"] = bool(
        dataset_stats.get("personalization_metrics", {})
        .get("unique_items_last_100", {})
        .get("avg", 0.0)
        < max(10, 0.3 * min(100, args.maxlen))
    )
    issues["stale_behavior"] = bool(recent_ratio is not None and recent_ratio < 0.2)
    issues["weak_ndcg"] = bool(ndcg is not None and ndcg < 0.01)
    return issues


def _generate_actionable_steps(
    issues: Mapping[str, bool],
    dataset_stats: dict,
    args: argparse.Namespace,
) -> List[str]:
    steps: List[str] = []
    if issues.get("too_long_sequences"):
        steps.append("Reduce max_seq_len to 150 and only keep the last 150 interactions per user (--maxlen 150).")
    if issues.get("popularity_collapse") or issues.get("poor_diversity"):
        steps.append(
            "Downsample the top 1% most popular items or apply popularity-aware reweighting to break popularity collapse."
        )
    if issues.get("insufficient_negative_sampling"):
        steps.append("Increase negative samples to 50–200 per step to sharpen InfoNCE discrimination.")
    if issues.get("underfitting"):
        steps.append("Train for 20–40 epochs and increase hidden_units to 128–256 for more capacity.")
    if issues.get("overfitting"):
        steps.append("Enable time-decay weighting or higher dropout to keep validation loss aligned with training loss.")
    if issues.get("low_personalization"):
        steps.append("Add user embeddings or switch to DuoRec for stronger personalization signals.")
    if issues.get("stale_behavior"):
        steps.append("Enable time-decay weighting so recent (<30d) events dominate the loss.")
    if dataset_stats.get("popularity_bias_score", 0.0) > 0.5:
        steps.append("Apply a popularity prior (e.g., log-frequency penalty) or diversity re-ranking after scoring.")
    if not steps:
        steps.append("Fine-tune learning rate and rerun for longer to push hit@10 higher.")
    return steps


def _compute_training_health(
    dataset_stats: dict,
    epoch_summaries: List[dict],
    best_epoch: dict,
    eval_metrics: Optional[dict],
    args: argparse.Namespace,
) -> dict:
    coverage = None
    ndcg_value = None
    if eval_metrics:
        coverage = eval_metrics.get("catalog_coverage")
        ndcg_value = eval_metrics.get("ndcg@k") or eval_metrics.get("ndcg")

    first_loss = epoch_summaries[0]["valid_loss"] if epoch_summaries else best_epoch["valid_loss"]
    best_loss = best_epoch["valid_loss"]
    improvement = max(0.0, first_loss - best_loss)
    loss_score = min(1.0, improvement / max(first_loss, 1e-8))
    hit_rate = best_epoch.get("valid_hit_rate@10", 0.0)
    hit_score = min(1.0, hit_rate / 0.2)
    ndcg_score = None
    if ndcg_value is not None:
        ndcg_score = min(1.0, ndcg_value / 0.2)

    pop_bias = dataset_stats.get("popularity_bias_score", 0.0)
    diversity_value = coverage if coverage is not None else (1.0 - pop_bias)
    diversity_score = min(1.0, max(0.0, diversity_value))

    unique_avg = (
        dataset_stats.get("personalization_metrics", {})
        .get("unique_items_last_100", {})
        .get("avg", 0.0)
    )
    denom = max(1.0, min(100.0, dataset_stats.get("avg_sequence_length", 1.0)))
    personalization_score = min(1.0, unique_avg / denom)
    if coverage is not None:
        personalization_score = min(1.0, 0.5 * personalization_score + 0.5 * coverage)

    long_tail_ratio = dataset_stats.get("long_tail_ratio", 0.0)
    long_tail_score = min(1.0, max(0.0, long_tail_ratio))
    popularity_component = min(1.0, max(0.0, 1.0 - pop_bias))

    components: Dict[str, Dict[str, float | None]] = {
        "loss_curve": {"value": improvement, "score": loss_score},
        "hit_rate": {"value": hit_rate, "score": hit_score},
        "ndcg": {"value": ndcg_value, "score": ndcg_score},
        "diversity": {"value": diversity_value, "score": diversity_score},
        "personalization": {"value": unique_avg, "score": personalization_score},
        "popularity_bias": {"value": pop_bias, "score": popularity_component},
        "long_tail_strength": {"value": long_tail_ratio, "score": long_tail_score},
    }

    score_values = [item["score"] for item in components.values() if item["score"] is not None]
    overall = sum(score_values) / len(score_values) * 100 if score_values else 0.0

    if overall >= 85:
        label = "Excellent"
    elif overall >= 70:
        label = "Good"
    elif overall >= 50:
        label = "Needs Work"
    elif overall >= 30:
        label = "Poor"
    else:
        label = "Critical"

    issues = _detect_training_issues(dataset_stats, best_epoch, epoch_summaries, eval_metrics, args)
    steps = _generate_actionable_steps(issues, dataset_stats, args)

    return {
        "score": overall,
        "label": label,
        "components": components,
        "issues": issues,
        "actionable_next_steps": steps,
        "best_epoch": best_epoch,
        "eval_metrics": eval_metrics,
    }


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KuaiRec baseline model.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", type=int, default=150)
    parser.add_argument("--hidden_units", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2_emb", type=float, default=0.0)
    parser.add_argument(
        "--neg_sampling_power",
        type=float,
        default=0.75,
        help="Exponent for popularity-weighted negative sampling (word2vec-style "
        "dampened frequency; 0 = uniform random negatives, 1 = raw frequency).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--state_dict_path", type=str, default=None)
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trim_sequences_to",
        type=int,
        default=150,
        help="Trim each user history to this many most recent interactions before training.",
    )
    parser.add_argument(
        "--head_downsample_percent",
        type=float,
        default=0.01,
        help="Top percentage of items considered 'head' for downsampling.",
    )
    parser.add_argument(
        "--head_downsample_keep_prob",
        type=float,
        default=0.3,
        help="Probability of keeping an interaction involving a head item.",
    )
    parser.add_argument(
        "--head_downsample_seed",
        type=int,
        default=42,
        help="Random seed for head-item downsampling.",
    )
    parser.add_argument(
        "--use_llm_emb",
        action="store_true",
        help="Use LLM-enhanced item (and optionally user) embeddings from kuairec/embeddings/.",
    )
    parser.add_argument(
        "--use_user_llm",
        action="store_true",
        help="Inject user LLM profile as a prefix token (requires --use_llm_emb).",
    )
    parser.add_argument(
        "--item_llm_path",
        type=str,
        default=None,
        help="Path to item_llm_embeddings.npy (default: kuairec/embeddings/item_llm_embeddings.npy).",
    )
    parser.add_argument(
        "--user_llm_path",
        type=str,
        default=None,
        help="Path to user_llm_embeddings.npy (default: kuairec/embeddings/user_llm_embeddings.npy).",
    )
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

    data_root = Path(os.environ.get("TRAIN_DATA_PATH", "./kuairec/data")).expanduser().resolve()
    data = load_kuairec_data(data_root)
    transformations: List[Dict[str, object]] = []
    if args.trim_sequences_to and args.trim_sequences_to > 0:
        data, trim_meta = trim_user_sequences(data, args.trim_sequences_to)
        trim_meta["type"] = "sequence_trim"
        transformations.append(trim_meta)

    if args.head_downsample_keep_prob < 1.0 and args.head_downsample_percent > 0:
        data, down_meta = downsample_head_items(
            data,
            head_percent=args.head_downsample_percent,
            keep_probability=args.head_downsample_keep_prob,
            seed=args.head_downsample_seed,
        )
        down_meta["type"] = "head_downsampling"
        transformations.append(down_meta)

    dataset_stats = compute_dataset_statistics(data)
    dataset_diag_payload = dict(dataset_stats)
    dataset_diag_payload["dataset_root"] = str(data_root)
    dataset_diag_payload["generated_at"] = time.time()
    dataset_diag_payload["transformations"] = transformations
    dataset_diag_path = ckpt_dir / "dataset_diagnostics.json"
    with dataset_diag_path.open("w", encoding="utf-8") as diag_file:
        json.dump(dataset_diag_payload, diag_file, indent=2)
    print(f"Dataset diagnostics saved to {dataset_diag_path}")
    dataset = KuaiRecTrainDataset(data, maxlen=args.maxlen, neg_sampling_power=args.neg_sampling_power)

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

    if args.use_llm_emb:
        import numpy as np
        emb_root = data_root.parent / "embeddings"
        item_llm_path = Path(args.item_llm_path) if args.item_llm_path else emb_root / "item_llm_embeddings.npy"
        print(f"Loading item LLM embeddings from: {item_llm_path}")
        item_llm = torch.tensor(np.load(item_llm_path), dtype=torch.float32)

        user_llm = None
        if args.use_user_llm:
            user_llm_path = Path(args.user_llm_path) if args.user_llm_path else emb_root / "user_llm_embeddings.npy"
            print(f"Loading user LLM embeddings from: {user_llm_path}")
            user_llm = torch.tensor(np.load(user_llm_path), dtype=torch.float32)

        model = LLMKuaiRecModel(
            num_items=data.num_items,
            hidden_units=args.hidden_units,
            maxlen=args.maxlen,
            num_heads=args.num_heads,
            num_blocks=args.num_blocks,
            dropout_rate=args.dropout_rate,
            item_llm_embeddings=item_llm,
            user_llm_embeddings=user_llm,
            num_users=data.num_users if user_llm is not None else None,
            norm_first=args.norm_first,
        ).to(device)
        print(f"Using LLMKuaiRecModel (user_prefix={'yes' if args.use_user_llm else 'no'})")
    else:
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
    epoch_summaries = []

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
            user = batch.get("user")
            user = user.to(device) if user is not None else None
            fwd_kwargs = {"user_ids": user} if args.use_llm_emb and args.use_user_llm else {}
            seq_repr, pos_embs, neg_embs = model(seq, pos, neg, **fwd_kwargs)
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
                user = batch.get("user")
                user = user.to(device) if user is not None else None
                fwd_kwargs = {"user_ids": user} if args.use_llm_emb and args.use_user_llm else {}
                seq_repr, pos_embs, neg_embs = model(seq, pos, neg, **fwd_kwargs)
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
        epoch_summaries.append(summary)
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
            "dataset_root": str(data_root),
            "dataset_statistics": dataset_stats,
            "timestamp": time.time(),
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2)

    print("Done")
    if epoch_summaries:
        best_epoch = min(epoch_summaries, key=lambda item: item["valid_loss"])
        eval_metrics = _load_eval_metrics()
        health_info = _compute_training_health(
            dataset_stats, epoch_summaries, best_epoch, eval_metrics, args
        )
        health_path = ckpt_dir / "training_health.json"
        with health_path.open("w", encoding="utf-8") as health_file:
            json.dump(health_info, health_file, indent=2)

        head_dist = dataset_stats.get("head_item_distribution", {})
        tail_dist = dataset_stats.get("tail_item_distribution", {})
        personalization = dataset_stats.get("personalization_metrics", {})
        unique_stats = personalization.get("unique_items_last_100", {})
        category_div = personalization.get("category_diversity_last_100")
        recent_ratio = personalization.get("recent_30d_interaction_ratio")

        summary_lines = [
            "KuaiRec Training Summary",
            f"Dataset directory: {data_root}",
            (
                f"Users: {dataset_stats['num_users']} | Items: {dataset_stats['num_items']} | "
                f"Interactions: {dataset_stats['total_interactions']}"
            ),
            "",
            "Sequence length statistics:",
            (
                f"  avg={dataset_stats['avg_sequence_length']:.2f} | "
                f"median={dataset_stats['median_sequence_length']:.2f} | "
                f"min={dataset_stats['min_sequence_length']} | "
                f"max={dataset_stats['max_sequence_length']}"
            ),
            "",
            "Head/tail distribution:",
            (
                "  Top 1% share: "
                f"{head_dist.get('top_1_percent', {}).get('share', 0.0):.2%} "
                f"({head_dist.get('top_1_percent', {}).get('count', 0)} items), "
                "Top 5% share: "
                f"{head_dist.get('top_5_percent', {}).get('share', 0.0):.2%}, "
                "Top 10% share: "
                f"{head_dist.get('top_10_percent', {}).get('share', 0.0):.2%}"
            ),
            (
                "  Tail <10/<5/<2 share: "
                f"{tail_dist.get('<10', {}).get('interaction_share', 0.0):.2%} / "
                f"{tail_dist.get('<5', {}).get('interaction_share', 0.0):.2%} / "
                f"{tail_dist.get('<2', {}).get('interaction_share', 0.0):.2%}"
            ),
            (
                "  Long-tail ratio (<10 freq): "
                f"{dataset_stats.get('long_tail_ratio', 0.0):.2%} | "
                f"Popularity bias score: {dataset_stats.get('popularity_bias_score', 0.0):.2f}"
            ),
            "",
            "Personalization snapshot:",
            (
                "  Unique items in last 100 (avg/median/min/max): "
                f"{unique_stats.get('avg', 0.0):.2f} / "
                f"{unique_stats.get('median', 0.0):.2f} / "
                f"{unique_stats.get('min', 0)} / "
                f"{unique_stats.get('max', 0)}"
            ),
            (
                "  Item entropy/Gini: "
                f"{personalization.get('item_entropy_bits', 0.0):.2f} bits / "
                f"{personalization.get('item_gini', 0.0):.2f}"
            ),
        ]
        summary_lines.append("")
        summary_lines.append("Data preprocessing steps:")
        if transformations:
            for transform in transformations:
                name = transform.get("type", "unknown")
                if name == "sequence_trim":
                    if transform.get("applied"):
                        summary_lines.append(
                            (
                                "  Trimmed sequences to last "
                                f"{transform.get('max_sequence_length')} interactions "
                                f"for {transform.get('affected_users', 0)} users ("
                                f"{transform.get('removed_interactions', 0)} interactions removed)."
                            )
                        )
                    else:
                        summary_lines.append(
                            "  Sequence trim requested but not needed (all sequences shorter than limit)."
                        )
                elif name == "head_downsampling":
                    if transform.get("applied"):
                        summary_lines.append(
                            (
                                "  Downsampled top "
                                f"{transform.get('head_percent', 0.0) * 100:.1f}% items "
                                f"(keep_prob={transform.get('keep_probability')}) removing "
                                f"{transform.get('removed_interactions', 0)} interactions across "
                                f"{transform.get('affected_users', 0)} users."
                            )
                        )
                    else:
                        summary_lines.append(
                            "  Head-item downsampling skipped (no qualifying head items or keep_prob>=1)."
                        )
                else:
                    summary_lines.append(f"  {name}: {transform}")
        else:
            summary_lines.append("  None (raw sequences used).")
        if category_div:
            summary_lines.append(
                (
                    "  Category diversity (avg/median): "
                    f"{category_div.get('avg', 0.0):.2f} / "
                    f"{category_div.get('median', 0.0):.2f}"
                )
            )
        if recent_ratio is not None:
            summary_lines.append(
                f"  Last-30-day interaction ratio: {recent_ratio:.2%}"
            )

        summary_lines.extend(
            [
                "",
                "Best epoch (by validation loss):",
                (
                    f"  epoch {best_epoch['epoch']} with valid_loss={best_epoch['valid_loss']:.4f}, "
                    f"hit_rate@10={best_epoch['valid_hit_rate@10']:.4f}, "
                    f"top1_acc={best_epoch['valid_top1_acc']:.4f}"
                ),
            ]
        )

        summary_lines.append("")
        summary_lines.append("Most interacted items:")
        for item in dataset_stats["top_items"][:5]:
            summary_lines.append(f"  {item['item_id']}: {item['count']} interactions")
        dominant_items = dataset_stats.get("dominant_items", [])
        summary_lines.append(
            "  Dominant items: "
            + (
                ", ".join(
                    f"{entry['item_id']} ({entry['user_coverage']:.0%} users)"
                    for entry in dominant_items[:5]
                )
                if dominant_items
                else "None detected"
            )
        )

        summary_lines.append("")
        summary_lines.append(
            f"Training health score: {health_info['score']:.1f}/100 ({health_info['label']})."
        )
        issue_list = [
            name.replace("_", " ")
            for name, active in health_info.get("issues", {}).items()
            if active
        ]
        summary_lines.append(
            "Issues detected: " + (", ".join(issue_list) if issue_list else "None")
        )

        if eval_metrics:
            summary_lines.append("")
            summary_lines.append("Evaluation snapshot (latest metrics.json):")
            summary_lines.append(
                (
                    f"  Hit rate@{eval_metrics.get('topk', 10)}: "
                    f"{eval_metrics.get('hit_rate@k', 0.0):.4f} | "
                    f"NDCG: {eval_metrics.get('ndcg@k', eval_metrics.get('ndcg', 0.0)):.4f} | "
                    f"Coverage: {eval_metrics.get('catalog_coverage', 0.0):.2%}"
                )
            )

        summary_lines.append("")
        summary_lines.append("Actionable next steps:")
        for step in health_info.get("actionable_next_steps", []):
            summary_lines.append(f"- {step}")

        summary_lines.append("")
        summary_lines.append(f"Dataset diagnostics: {dataset_diag_path}")
        summary_lines.append(f"Training health JSON: {health_path}")

        summary_text = "\n".join(summary_lines) + "\n"
        summary_path = ckpt_dir / "latest_training_summary.txt"
        with summary_path.open("w", encoding="utf-8") as summary_file:
            summary_file.write(summary_text)
        print(summary_text)

    writer.close()
    log_file.close()
