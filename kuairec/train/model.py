"""Simplified KuaiRec baseline model that is self-contained within the package."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KuaiRecModel(nn.Module):
    """Transformer-based sequential recommender for KuaiRec."""

    def __init__(
        self,
        num_items: int,
        hidden_units: int,
        maxlen: int,
        num_heads: int,
        num_blocks: int,
        dropout_rate: float,
        norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.hidden_units = hidden_units
        self.maxlen = maxlen

        self.item_embedding = nn.Embedding(num_items + 1, hidden_units, padding_idx=0)
        self.position_embedding = nn.Embedding(maxlen, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units,
            nhead=num_heads,
            dim_feedforward=hidden_units * 4,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-8)
        self.temp = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        seq_items: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return encoded sequence representations and item embeddings."""

        seq_repr = self.encode_sequence(seq_items)
        pos_embs = self.item_embedding(pos_items)
        neg_embs = self.item_embedding(neg_items)
        return seq_repr, pos_embs, neg_embs

    def encode_sequence(self, seq_items: torch.Tensor) -> torch.Tensor:
        """Encode item histories into contextualised representations."""

        batch_size, seq_len = seq_items.shape
        positions = (
            torch.arange(seq_len, device=seq_items.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        item_embs = self.item_embedding(seq_items)
        pos_embs = self.position_embedding(positions)
        x = self.dropout(item_embs + pos_embs)
        padding_mask = seq_items == 0
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.layer_norm(encoded)

    def compute_infonce_loss(
        self,
        seq_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor,
        loss_mask: torch.Tensor,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]] | torch.Tensor:
        """Compute an InfoNCE loss with optional retrieval metrics."""

        hidden_size = seq_embs.size(-1)
        mask = loss_mask.bool()
        if mask.sum() == 0:
            zero_loss = seq_embs.sum() * 0
            metrics = {"total": 0, "correct": 0, "hit_at_5": 0, "hit_at_10": 0}
            if return_metrics:
                return zero_loss, metrics
            return zero_loss

        seq_valid = seq_embs[mask]
        pos_valid = pos_embs[mask]

        seq_valid = F.normalize(seq_valid, dim=-1)
        pos_valid = F.normalize(pos_valid, dim=-1)

        pos_logits = torch.sum(seq_valid * pos_valid, dim=-1, keepdim=True)
        neg_all = neg_embs.reshape(-1, hidden_size)
        neg_all = F.normalize(neg_all, dim=-1)
        neg_logits = torch.matmul(seq_valid, neg_all.transpose(-1, -2))

        logits = torch.cat([pos_logits, neg_logits], dim=-1) / self.temp
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels)

        if not return_metrics:
            return loss

        with torch.no_grad():
            total = logits.size(0)
            preds = logits.argmax(dim=1)
            correct = (preds == 0).sum().item()
            max_k = min(10, logits.size(1))
            topk = logits.topk(max_k, dim=1).indices
            hit_at_5 = (topk[:, : min(5, max_k)] == 0).any(dim=1).sum().item()
            hit_at_10 = (topk == 0).any(dim=1).sum().item()

        metrics = {
            "total": total,
            "correct": correct,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10,
        }
        return loss, metrics


__all__ = ["KuaiRecModel"]
