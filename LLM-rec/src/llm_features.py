# LLM-rec/src/llm_features.py
import numpy as np
import torch


def load_item_llm_embeddings(path: str) -> torch.Tensor:
    """
    Load precomputed item LLM embeddings from a .npy file.
    Shape: [n_items, llm_embedding_dim].
    """
    arr = np.load(path)
    return torch.tensor(arr, dtype=torch.float32)
