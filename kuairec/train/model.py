"""KuaiRec-specific model helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from train.model import BaselineModel as _BaselineModel


class KuaiRecBaselineModel(_BaselineModel):
    """Extension of the baseline model with optional retrieval metrics."""

    def compute_infonce_loss(
        self,
        seq_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor,
        loss_mask: torch.Tensor,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]] | torch.Tensor:
        hidden_size = neg_embs.size(-1)

        # L2-normalize embeddings
        seq_embs = torch.nn.functional.normalize(seq_embs, dim=-1)
        pos_embs = torch.nn.functional.normalize(pos_embs, dim=-1)
        neg_embs = torch.nn.functional.normalize(neg_embs, dim=-1)

        # positive logits: cosine similarity
        pos_logits = torch.nn.functional.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)

        # in-batch negatives: contrast each position against ALL negatives in the batch
        neg_embedding_all = neg_embs.reshape(-1, hidden_size)
        neg_logits = torch.matmul(seq_embs, neg_embedding_all.transpose(-1, -2))

        # concatenate [pos | negs]
        logits = torch.cat([pos_logits, neg_logits], dim=-1)

        # keep only valid (next item) positions
        valid_logits = logits[loss_mask.bool()]

        if valid_logits.size(0) == 0:
            zero_loss = seq_embs.sum() * 0
            metrics = {
                "total": 0,
                "correct": 0,
                "hit_at_5": 0,
                "hit_at_10": 0,
            }
            if return_metrics:
                return zero_loss, metrics
            return zero_loss

        # apply temperature
        valid_logits = valid_logits / self.temp

        # labels: 0 means the first column (the positive) is the correct class
        labels = torch.zeros(valid_logits.size(0), device=valid_logits.device, dtype=torch.int64)

        # cross-entropy over 1 + (B*L) classes
        loss = torch.nn.functional.cross_entropy(valid_logits, labels)

        if not return_metrics:
            return loss

        with torch.no_grad():
            total = valid_logits.size(0)
            preds = valid_logits.argmax(dim=1)
            correct = (preds == 0).sum().item()

            max_k = min(10, valid_logits.size(1))
            if max_k <= 0:
                hit_at_5 = 0
                hit_at_10 = 0
            else:
                topk_indices = valid_logits.topk(max_k, dim=1).indices
                top5_slice = topk_indices[:, : min(5, max_k)]
                hit_at_5 = (top5_slice == 0).any(dim=1).sum().item()
                hit_at_10 = (topk_indices == 0).any(dim=1).sum().item()

        metrics = {
            "total": total,
            "correct": correct,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10,
        }

        return loss, metrics


__all__ = ["KuaiRecBaselineModel"]
