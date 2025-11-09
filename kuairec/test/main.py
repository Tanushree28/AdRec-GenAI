"""KuaiRec inference entrypoint independent of the Tencent baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kuairec.train.dataset import (
    KuaiRecEvalDataset,
    is_valid_kuairec_root,
    load_kuairec_data,
)
from kuairec.train.model import KuaiRecModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "kuairec" / "ckpt"


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
    env_dataset_root = _env_path("EVAL_DATA_PATH", "TRAIN_DATA_PATH") or Path("kuairec/data")
    env_result_dir = _env_path("EVAL_RESULT_PATH") or Path("kuairec/eval_results")
    env_checkpoint = _env_path(
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
    parser.add_argument("--maxlen", type=int, default=101)
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
            f"{dataset_root}. Provide the correct directory via --dataset-root or EVAL_DATA_PATH. "
            "On Windows PowerShell use `$env:EVAL_DATA_PATH=...`; in Command Prompt use `set EVAL_DATA_PATH=...`."
        )

    data = load_kuairec_data(dataset_root)
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

            for user_idx, rec_indices in zip(users, topk_indices.tolist()):
                user_id = user_inverse.get(int(user_idx), str(user_idx))
                rec_items = [item_inverse.get(int(item), str(item)) for item in rec_indices]
                record = {"user_id": user_id, "recommendations": rec_items}
                output_file.write(json.dumps(record) + "\n")

    hit_rate = metrics_hits / total_users if total_users else 0.0
    ndcg = metrics_ndcg / total_users if total_users else 0.0

    metrics = {
        "users_evaluated": total_users,
        "topk": args.topk,
        "hit_rate@k": hit_rate,
        "ndcg@k": ndcg,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(
        f"Evaluation complete: hit_rate@{args.topk}={hit_rate:.4f}, "
        f"ndcg@{args.topk}={ndcg:.4f}."
    )
    print(f"Recommendations saved to {output_path}")
    print(f"Metrics saved to {result_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
