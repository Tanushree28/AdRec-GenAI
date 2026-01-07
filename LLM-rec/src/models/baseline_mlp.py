# LLM-rec/src/models/baseline_mlp.py
import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_embedding_dim: int,
        item_embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, user_embedding_dim)
        self.item_emb = nn.Embedding(n_items, item_embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(user_embedding_dim + item_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        x = torch.cat([u, v], dim=-1)
        out = self.mlp(x).squeeze(-1)
        return out
