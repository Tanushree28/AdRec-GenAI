"""KuaiRec inference entrypoint independent of the Tencent baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kuairec.train.dataset import (
    KuaiRecEvalDataset,
    compute_dataset_statistics,
    entropy_from_counts,
    gini_from_counts,
    is_valid_kuairec_root,
    load_kuairec_data,
)
from kuairec.train.model import KuaiRecModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "kuairec" / "ckpt"


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _diagnose_recommendations(
    hit_rate: float,
    ndcg: float,
    coverage: float,
    zero_hit_rate: float,
    rec_entropy: float,
    rec_gini: float,
    dataset_stats: dict,
    unique_recommended_ratio: float,
    args: argparse.Namespace,
) -> dict:
    issues = {}
    avg_len = dataset_stats.get("avg_sequence_length", 0.0)
    long_tail_ratio = dataset_stats.get("long_tail_ratio", 0.0)
    pop_bias = dataset_stats.get("popularity_bias_score", 0.0)
    recent_ratio = (
        dataset_stats.get("personalization_metrics", {})
        .get("recent_30d_interaction_ratio")
    )

    issues["popularity_collapse"] = bool(
        coverage < 0.35 or rec_gini > 0.9 or rec_entropy < 5.0
    )
    issues["underfitting"] = bool(hit_rate < 0.02 or ndcg < 0.01)
    issues["poor_diversity"] = bool(coverage < 0.4 or rec_entropy < 6.0)
    issues["too_long_sequences"] = bool(avg_len > args.maxlen * 1.5)
    issues["insufficient_negative_sampling"] = bool(hit_rate < 0.02 and long_tail_ratio > 0.3)
    issues["personalization_gap"] = bool(zero_hit_rate > 0.9 or unique_recommended_ratio < 0.3)
    issues["stale_behavior"] = bool(recent_ratio is not None and recent_ratio < 0.2)
    issues["popularity_bias"] = bool(pop_bias > 0.6)
    return issues


def _recommendation_next_steps(issues: Mapping[str, bool]) -> list[str]:
    steps: list[str] = []
    if issues.get("too_long_sequences"):
        steps.append("Reduce max_seq_len to 150 to focus on the freshest 150 interactions per user.")
    if issues.get("insufficient_negative_sampling"):
        steps.append("Increase negative samples to 50–200 per batch so InfoNCE can distinguish positives.")
    if issues.get("popularity_collapse") or issues.get("poor_diversity"):
        steps.append("Downsample the top 1% items and add diversity-promoting re-ranking to avoid repeating the same ads.")
    if issues.get("personalization_gap"):
        steps.append("Add user embeddings or switch to DuoRec for personalization and apply user-level regularization.")
    if issues.get("underfitting"):
        steps.append("Train for 20–40 epochs with hidden_units=128–256 to raise hit@10 and NDCG.")
    if issues.get("stale_behavior"):
        steps.append("Enable time-decay weighting so interactions from the last 30 days dominate training.")
    if issues.get("popularity_bias"):
        steps.append("Apply popularity-aware loss or log-frequency penalties before sampling negatives.")
    if not steps:
        steps.append("Fine-tune learning rate and refresh checkpoints to push evaluation metrics higher.")
    return steps


def _env_path(*names: str) -> Path | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(p for p in path.iterdir() if p.suffix == ".pt")
        if not candidates:
            raise FileNotFoundError(f"No .pt file found in checkpoint directory {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def _discover_latest_checkpoint(base: Path = DEFAULT_CKPT_ROOT) -> Path | None:
    base = base.expanduser().resolve()
    if not base.exists():
        return None

    candidates = []
    try:
        for path in base.rglob("*.pt"):
            if path.is_file():
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, path))
    except OSError:
        return None

    if not candidates:
        return None

    candidates.sort()
    return candidates[-1][1]


def _looks_like_tencent_state(state_dict: Mapping[str, object]) -> bool:
    suspicious_prefixes = (
        "item_emb",
        "user_emb",
        "sparse_emb",
        "attention_layers",
        "forward_layers",
    )
    for key in state_dict.keys():
        stripped = key.split(".", 1)[1] if key.startswith("module.") else key
        if stripped.startswith(suspicious_prefixes):
            return True
    return False


def main() -> int:
    env_dataset_root = _env_path(
        "EVAL_REC_DATA_PATH",
        "TRAIN_REC_DATA_PATH",
        "EVAL_DATA_PATH",
        "TRAIN_DATA_PATH",
    ) or Path("kuairec/data")
    env_result_dir = _env_path("EVAL_REC_RESULT_PATH", "EVAL_RESULT_PATH") or Path(
        "kuairec/eval_results"
    )
    env_checkpoint = _env_path(
        "MODEL_REC_OUTPUT_PATH",
        "EVAL_REC_MODEL_PATH",
        "EVAL_REC_CHECKPOINT_PATH",
        "TRAIN_REC_CKPT_PATH",
        "MODEL_OUTPUT_PATH",
        "EVAL_MODEL_PATH",
        "EVAL_CHECKPOINT_PATH",
        "TRAIN_CKPT_PATH",
    )

    parser = argparse.ArgumentParser(description="Run KuaiRec inference.")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=150)
    parser.add_argument("--hidden-units", dest="hidden_units", type=int, default=32)
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=1)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=1)
    parser.add_argument("--dropout-rate", dest="dropout_rate", type=float, default=0.2)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    user_supplied_root = args.dataset_root is not None
    dataset_root = (args.dataset_root or env_dataset_root).expanduser().resolve()
    result_dir = (args.result_dir or env_result_dir).expanduser().resolve()
    checkpoint_value = args.checkpoint or env_checkpoint
    if checkpoint_value is None:
        checkpoint_value = _discover_latest_checkpoint()

    if checkpoint_value is None:
        parser.error(
            "Unable to resolve a checkpoint. Pass --checkpoint, set MODEL_OUTPUT_PATH/EVAL_CHECKPOINT_PATH, "
            "or place KuaiRec checkpoints under kuairec/ckpt."
        )

    checkpoint = _resolve_checkpoint(checkpoint_value.expanduser().resolve())

    metadata_path = checkpoint.parent / "metadata.json"
    metadata: dict | None = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        saved_args = metadata.get("args", {})
        for key in ["batch_size", "maxlen", "hidden_units", "num_blocks", "num_heads", "dropout_rate"]:
            if key in saved_args:
                setattr(args, key, saved_args[key])
        if "norm_first" in saved_args:
            args.norm_first = bool(saved_args["norm_first"])

        saved_root = metadata.get("dataset_root")
        if saved_root and not user_supplied_root:
            saved_root_path = Path(saved_root)
            if is_valid_kuairec_root(saved_root_path):
                dataset_root = saved_root_path

    if not is_valid_kuairec_root(dataset_root):
        raise FileNotFoundError(
            "KuaiRec dataset not found or missing small_matrix.csv/big_matrix.csv at "
            f"{dataset_root}. Provide the correct directory via --dataset-root or EVAL_REC_DATA_PATH. "
            "On Windows PowerShell use `$env:EVAL_REC_DATA_PATH=...`; in Command Prompt use `set EVAL_REC_DATA_PATH=...`."
        )

    data = load_kuairec_data(dataset_root)
    dataset_stats = compute_dataset_statistics(data)
    eval_dataset = KuaiRecEvalDataset(data, maxlen=args.maxlen)
    if len(eval_dataset) == 0:
        raise ValueError("Evaluation dataset is empty. Each user must have at least two interactions.")

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=KuaiRecEvalDataset.collate_fn,
    )

    result_dir.mkdir(parents=True, exist_ok=True)

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

    state_dict = torch.load(checkpoint, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:  # pragma: no cover - defensive error path
        if _looks_like_tencent_state(state_dict):
            raise RuntimeError(
                "The provided checkpoint appears to come from the Tencent baseline. "
                "Run kuairec.train.run to produce KuaiRec-specific checkpoints before evaluating."
            ) from exc
        raise
    model.eval()

    item_embeddings = model.item_embedding.weight.detach().to(device)
    item_embeddings = F.normalize(item_embeddings, dim=-1)
    user_inverse = data.user_inverse
    item_inverse = data.item_inverse

    metrics_hits = 0
    metrics_ndcg = 0.0
    total_users = 0
    hit_ranks: list[float] = []
    recommended_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    output_path = result_dir / "recommendations.jsonl"

    with torch.no_grad(), output_path.open("w", encoding="utf-8") as output_file:
        for batch in tqdm(eval_loader, total=len(eval_loader)):
            seq = batch["seq"].to(device)
            target = batch["target"].to(device)
            lengths = batch["length"].to(device)
            users = batch["user"].tolist()
            history_items = batch["history_items"]

            encoded = model.encode_sequence(seq)
            positions = lengths - 1
            batch_indices = torch.arange(seq.size(0), device=device)
            user_repr = encoded[batch_indices, positions]
            user_repr = F.normalize(user_repr, dim=-1)

            scores = torch.matmul(user_repr, item_embeddings.T)
            scores[:, 0] = float("-inf")
            for row, seen in enumerate(history_items):
                if not seen:
                    continue
                seen_tensor = torch.tensor(list(seen), device=device, dtype=torch.long)
                scores[row, seen_tensor] = float("-inf")

            topk_indices = scores.topk(args.topk, dim=1).indices

            total_users += target.size(0)
            hits_mask = topk_indices == target.view(-1, 1)
            hit_rows = hits_mask.any(dim=1)
            metrics_hits += hit_rows.sum().item()

            match_positions = hits_mask.float().argmax(dim=1)
            for idx, is_hit in enumerate(hit_rows):
                if is_hit:
                    rank = match_positions[idx].item()
                    metrics_ndcg += 1.0 / math.log2(rank + 2)
                    hit_ranks.append(rank + 1)

            for user_idx, rec_indices in zip(users, topk_indices.tolist()):
                user_id = user_inverse.get(int(user_idx), str(user_idx))
                rec_items = [item_inverse.get(int(item), str(item)) for item in rec_indices]
                recommended_counter.update(rec_items)
                record = {"user_id": user_id, "recommendations": rec_items}
                output_file.write(json.dumps(record) + "\n")

            for tgt in target.tolist():
                item_name = item_inverse.get(int(tgt), str(tgt))
                target_counter[item_name] += 1

    hit_rate = metrics_hits / total_users if total_users else 0.0
    ndcg = metrics_ndcg / total_users if total_users else 0.0
    unique_recommended = len(recommended_counter)
    coverage = (
        unique_recommended / dataset_stats["num_items"]
        if dataset_stats["num_items"]
        else 0.0
    )
    avg_hit_rank = sum(hit_ranks) / len(hit_ranks) if hit_ranks else None
    zero_hit_rate = 1.0 - (metrics_hits / total_users) if total_users else 1.0
    rec_entropy = entropy_from_counts(recommended_counter.values())
    rec_gini = gini_from_counts(recommended_counter.values())
    unique_ratio = coverage

    metrics = {
        "users_evaluated": total_users,
        "topk": args.topk,
        "hit_rate@k": hit_rate,
        "ndcg@k": ndcg,
        "hit_users": metrics_hits,
        "unique_recommended_items": unique_recommended,
        "catalog_coverage": coverage,
        "average_hit_rank": avg_hit_rank,
        "zero_hit_rate": zero_hit_rate,
        "recommendation_entropy_bits": rec_entropy,
        "recommendation_gini": rec_gini,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "dataset_statistics": dataset_stats,
        "most_common_recommendations": recommended_counter.most_common(10),
        "most_common_targets": target_counter.most_common(10),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    recommendation_issues = _diagnose_recommendations(
        hit_rate,
        ndcg,
        coverage,
        zero_hit_rate,
        rec_entropy,
        rec_gini,
        dataset_stats,
        unique_ratio,
        args,
    )
    recommendation_actions = _recommendation_next_steps(recommendation_issues)
    recommendation_diag = {
        "hit_rate@k": hit_rate,
        "ndcg@k": ndcg,
        "catalog_coverage": coverage,
        "zero_hit_rate": zero_hit_rate,
        "average_hit_rank": avg_hit_rank,
        "recommendation_entropy_bits": rec_entropy,
        "recommendation_gini": rec_gini,
        "unique_recommended_ratio": unique_ratio,
        "issues": recommendation_issues,
        "actionable_next_steps": recommendation_actions,
        "top_recommendations": recommended_counter.most_common(10),
        "top_ground_truth": target_counter.most_common(10),
    }
    recommendation_diag_path = result_dir / "recommendation_diagnostics.json"
    with recommendation_diag_path.open("w", encoding="utf-8") as diag_file:
        json.dump(recommendation_diag, diag_file, indent=2)

    summary_lines = [
        "KuaiRec Evaluation Summary",
        f"Dataset directory: {dataset_root}",
        (
            f"Users evaluated: {total_users} | Hits: {metrics_hits} | "
            f"Top-{args.topk} hit rate: {hit_rate:.4f}"
        ),
        f"Top-{args.topk} NDCG: {ndcg:.4f}",
        (
            f"Unique items recommended: {unique_recommended} (coverage {coverage:.2%} of catalog)"
        ),
        f"Zero-hit users: {zero_hit_rate:.2%}",
        (
            f"Recommendation entropy/Gini: {rec_entropy:.2f} bits / {rec_gini:.2f}"
        ),
    ]
    if avg_hit_rank is not None:
        summary_lines.append(f"Average rank when correct item found: {avg_hit_rank:.2f}")

    summary_lines.append("")
    summary_lines.append("Most recommended items (top 10):")
    for item, count in recommended_counter.most_common(10):
        summary_lines.append(f"  {item}: recommended {count} times")

    summary_lines.append("")
    summary_lines.append("Most common ground-truth targets (top 5):")
    for item, count in target_counter.most_common(5):
        summary_lines.append(f"  {item}: appeared {count} times")

    summary_lines.append("")
    summary_lines.append(
        "Sequence length stats (avg/median/min/max): "
        f"{dataset_stats['avg_sequence_length']:.2f} / "
        f"{dataset_stats['median_sequence_length']:.2f} / "
        f"{dataset_stats['min_sequence_length']} / "
        f"{dataset_stats['max_sequence_length']}"
    )

    summary_lines.append("")
    summary_lines.append("Recommendation issues detected: " + (
        ", ".join(name.replace("_", " ") for name, flag in recommendation_issues.items() if flag) or "None"
    ))

    summary_lines.append("")
    summary_lines.append("Actionable next steps:")
    for step in recommendation_actions:
        summary_lines.append(f"- {step}")

    summary_lines.append("")
    summary_lines.append(f"Recommendation diagnostics JSON: {recommendation_diag_path}")

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = result_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(summary_text)

    print(
        f"Evaluation complete: hit_rate@{args.topk}={hit_rate:.4f}, "
        f"ndcg@{args.topk}={ndcg:.4f}."
    )
    print(f"Recommendations saved to {output_path}")
    print(f"Metrics saved to {result_dir / 'metrics.json'}")
    print(f"Recommendation diagnostics saved to {recommendation_diag_path}")
    print(f"Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
