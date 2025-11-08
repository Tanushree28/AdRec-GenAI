"""KuaiRec inference entrypoint independent of the Tencent baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from kuairec.train.dataset import KuaiRecEvalDataset, load_kuairec_data
from kuairec.train.model import KuaiRecModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KuaiRec inference.")
    parser.add_argument("--dataset-root", type=Path, default=Path("kuairec/data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=Path("kuairec/eval_results"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=101)
    parser.add_argument("--hidden-units", dest="hidden_units", type=int, default=32)
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=1)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=1)
    parser.add_argument("--dropout-rate", dest="dropout_rate", type=float, default=0.2)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(p for p in path.iterdir() if p.suffix == ".pt")
        if not candidates:
            raise FileNotFoundError(f"No .pt file found in checkpoint directory {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def main() -> int:
    args = _parse_args()

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    data = load_kuairec_data(args.dataset_root)
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

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
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
        "checkpoint": str(checkpoint_path),
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
