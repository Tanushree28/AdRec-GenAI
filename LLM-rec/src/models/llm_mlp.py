# LLM-rec/src/models/llm_mlp.py
import torch
import torch.nn as nn


class LLMEnhancedMLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        llm_embedding_dim: int,
        hidden_dim: int,
        item_llm_embeddings: torch.Tensor,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, user_embedding_dim)
        self.item_emb = nn.Embedding(n_items, item_embedding_dim)

        # Frozen LLM embeddings as a lookup table
        assert item_llm_embeddings.shape[0] == n_items, "n_items mismatch"
        self.register_buffer("item_llm", item_llm_embeddings)

        input_dim = user_embedding_dim + item_embedding_dim + llm_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v_id = self.item_emb(item_ids)
        v_llm = self.item_llm[item_ids]  # [batch_size, llm_dim]
        x = torch.cat([u, v_id, v_llm], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out
