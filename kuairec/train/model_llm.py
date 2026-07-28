"""LLM-enhanced KuaiRec model.

Extends KuaiRecModel so that each item's input token is formed by
projecting the concatenation of the ID embedding and the frozen
LLM embedding down to ``hidden_units``.

Optionally, a user-level LLM embedding is prepended to the sequence
as a special [USER] prefix token.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import KuaiRecModel


class LLMKuaiRecModel(KuaiRecModel):
    """KuaiRec Transformer augmented with frozen LLM item (and optionally user) embeddings."""

    def __init__(
        self,
        num_items: int,
        hidden_units: int,
        maxlen: int,
        num_heads: int,
        num_blocks: int,
        dropout_rate: float,
        item_llm_embeddings: torch.Tensor,
        user_llm_embeddings: Optional[torch.Tensor] = None,
        num_users: Optional[int] = None,
        norm_first: bool = False,
    ) -> None:
        super().__init__(
            num_items=num_items,
            hidden_units=hidden_units,
            maxlen=maxlen,
            num_heads=num_heads,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            norm_first=norm_first,
        )

        # item_llm_embeddings.npy is indexed by the RAW 0-based item id (row i =
        # item whose raw id is i), but this model's item ids are reindexed 1..num_items
        # with 0 reserved for PAD (see kuairec/train/dataset.py: reid = idx + 1).
        # So row i of the raw array belongs at buffer index i + 1, not index i —
        # prepend a zero PAD row rather than appending, or every id is off by one
        # and the highest-id item silently gets no embedding at all.
        item_llm_embeddings = self._align_to_reindexed_ids(item_llm_embeddings, num_items)
        self.register_buffer("item_llm", item_llm_embeddings)
        llm_dim = item_llm_embeddings.shape[1]

        # Project [id_emb || llm_emb] → hidden_units
        self.item_proj = nn.Linear(hidden_units + llm_dim, hidden_units, bias=False)

        # Optional user LLM prefix token
        self.use_user_llm = user_llm_embeddings is not None
        if self.use_user_llm:
            if num_users is None:
                raise ValueError("num_users is required when user_llm_embeddings is provided")
            # Same raw-id -> reindexed-id (+1) alignment issue as item_llm above.
            user_llm_embeddings = self._align_to_reindexed_ids(user_llm_embeddings, num_users)
            user_llm_dim = user_llm_embeddings.shape[1]
            self.register_buffer("user_llm", user_llm_embeddings)
            self.user_llm_proj = nn.Linear(user_llm_dim, hidden_units, bias=False)
            # Position embedding has maxlen slots; we need one extra for the prefix
            self.prefix_position_emb = nn.Embedding(1, hidden_units)
        else:
            self.user_llm = None

    @staticmethod
    def _align_to_reindexed_ids(embeddings: torch.Tensor, num_ids: int) -> torch.Tensor:
        """Shift a raw-0-indexed embedding table onto reid space (PAD=0, ids 1..num_ids)."""
        n_rows = embeddings.shape[0]
        dim = embeddings.shape[1]
        if n_rows == num_ids:
            pad_row = torch.zeros(1, dim, dtype=embeddings.dtype)
            embeddings = torch.cat([pad_row, embeddings], dim=0)
        elif n_rows < num_ids + 1:
            pad = torch.zeros(num_ids + 1 - n_rows, dim, dtype=embeddings.dtype)
            embeddings = torch.cat([embeddings, pad], dim=0)
        elif n_rows > num_ids + 1:
            embeddings = embeddings[: num_ids + 1]
        return embeddings

    # ------------------------------------------------------------------
    # Override encode_sequence to fuse LLM embeddings
    # ------------------------------------------------------------------

    def encode_sequence(
        self,
        seq_items: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = seq_items.shape
        positions = (
            torch.arange(seq_len, device=seq_items.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # Item token = project(concat(id_emb, llm_emb))
        id_embs = self.item_embedding(seq_items)               # [B, L, H]
        llm_embs = self.item_llm[seq_items]                    # [B, L, llm_dim]
        fused = self.item_proj(torch.cat([id_embs, llm_embs], dim=-1))  # [B, L, H]

        pos_embs = self.position_embedding(positions)          # [B, L, H]
        x = self.dropout(fused + pos_embs)
        padding_mask = seq_items == 0                          # [B, L]

        if self.use_user_llm and user_ids is not None:
            # Build prefix token from user LLM embedding
            u_llm = self.user_llm[user_ids]                    # [B, user_llm_dim]
            prefix = self.user_llm_proj(u_llm).unsqueeze(1)    # [B, 1, H]
            prefix_pos = self.prefix_position_emb(
                torch.zeros(batch_size, 1, device=seq_items.device, dtype=torch.long)
            )                                                   # [B, 1, H]
            prefix = self.dropout(prefix + prefix_pos)
            # Prepend prefix; it's never padding
            prefix_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=seq_items.device)
            x = torch.cat([prefix, x], dim=1)                  # [B, L+1, H]
            padding_mask = torch.cat([prefix_mask, padding_mask], dim=1)  # [B, L+1]

        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.layer_norm(encoded)

    # ------------------------------------------------------------------
    # Override forward to pass user_ids through
    # ------------------------------------------------------------------

    def forward(
        self,
        seq_items: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        user_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_repr = self.encode_sequence(seq_items, user_ids=user_ids)

        # When the user-LLM prefix token was prepended, seq_repr has shape
        # [B, L+1, H]. Strip the prefix so the output matches the original
        # [B, L, H] that compute_infonce_loss (and its mask) expects.
        if self.use_user_llm and user_ids is not None:
            seq_repr = seq_repr[:, 1:, :]   # drop prefix token, keep item positions

        pos_embs = self._item_token(pos_items)
        neg_embs = self._item_token(neg_items)
        return seq_repr, pos_embs, neg_embs

    def _item_token(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return projected LLM-fused embedding for a batch of item ids."""
        id_e = self.item_embedding(item_ids)
        llm_e = self.item_llm[item_ids]
        return self.item_proj(torch.cat([id_e, llm_e], dim=-1))


__all__ = ["LLMKuaiRecModel"]
