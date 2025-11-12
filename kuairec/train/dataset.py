"""Utilities for loading KuaiRec data independent of the Tencent baseline."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class KuaiRecData:
    """Container for KuaiRec interaction data after reindexing."""

    user_sequences: Mapping[int, List[int]]
    user_map: Mapping[str, int]
    item_map: Mapping[str, int]

    @property
    def num_users(self) -> int:
        return len(self.user_map)

    @property
    def num_items(self) -> int:
        return len(self.item_map)

    @property
    def user_inverse(self) -> Dict[int, str]:
        return {v: k for k, v in self.user_map.items()}

    @property
    def item_inverse(self) -> Dict[int, str]:
        return {v: k for k, v in self.item_map.items()}


def compute_dataset_statistics(data: "KuaiRecData") -> Dict[str, object]:
    """Return aggregate statistics that make KuaiRec runs easier to interpret."""

    lengths = [len(seq) for seq in data.user_sequences.values()]
    total_interactions = int(sum(lengths))
    min_len = int(min(lengths)) if lengths else 0
    max_len = int(max(lengths)) if lengths else 0
    avg_len = float(mean(lengths)) if lengths else 0.0
    median_len = float(median(lengths)) if lengths else 0.0

    item_counts: Counter[int] = Counter()
    for sequence in data.user_sequences.values():
        item_counts.update(sequence)

    item_inverse = data.item_inverse
    top_items = [
        {
            "item_id": item_inverse.get(item, str(item)),
            "count": int(count),
        }
        for item, count in item_counts.most_common(20)
    ]

    return {
        "num_users": data.num_users,
        "num_items": data.num_items,
        "total_interactions": total_interactions,
        "avg_sequence_length": avg_len,
        "median_sequence_length": median_len,
        "min_sequence_length": min_len,
        "max_sequence_length": max_len,
        "top_items": top_items,
    }


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        for column in columns:
            if candidate in column.lower():
                return column
    return None


def is_valid_kuairec_root(path: str | Path) -> bool:
    """Return True if the directory looks like an extracted KuaiRec dataset."""

    data_path = Path(path)
    if not data_path.exists():
        return False
    for candidate in ("small_matrix.csv", "big_matrix.csv"):
        if (data_path / candidate).exists():
            return True
    return False


def load_kuairec_data(data_dir: str | Path) -> KuaiRecData:
    """Load KuaiRec CSVs into reindexed interaction sequences."""

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"KuaiRec data directory does not exist: {data_path}")

    interaction_path = data_path / "small_matrix.csv"
    if not interaction_path.exists():
        interaction_path = data_path / "big_matrix.csv"
    if not interaction_path.exists():
        available = ", ".join(sorted(p.name for p in data_path.glob("*.csv"))) or "<none>"
        raise FileNotFoundError(
            "Could not locate small_matrix.csv or big_matrix.csv in the KuaiRec data folder. "
            f"Found: {available} in {data_path}."
        )

    interactions = pd.read_csv(interaction_path)
    if interactions.empty:
        raise ValueError(f"No interactions found in {interaction_path}.")

    user_col = _find_column(interactions.columns, ["user_id", "userid", "uid"])
    item_col = _find_column(interactions.columns, ["item_id", "video_id", "iid", "cid"])
    if user_col is None or item_col is None:
        raise ValueError(
            "Unable to identify user/item columns in the KuaiRec interactions CSV."
        )

    time_col = _find_column(
        interactions.columns,
        ["timestamp", "time", "datetime", "ts", "datatime"],
    )

    interactions = interactions.dropna(subset=[user_col, item_col])
    interactions[user_col] = interactions[user_col].astype(str)
    interactions[item_col] = interactions[item_col].astype(str)

    if time_col is not None and time_col in interactions.columns:
        interactions = interactions.sort_values(time_col)

    user_ids = sorted(interactions[user_col].unique())
    item_ids = sorted(interactions[item_col].unique())

    user_map = {user_id: idx + 1 for idx, user_id in enumerate(user_ids)}
    item_map = {item_id: idx + 1 for idx, item_id in enumerate(item_ids)}

    user_sequences: Dict[int, List[int]] = {}
    grouped = interactions.groupby(user_col)
    for user_id, group in grouped:
        reid = user_map[user_id]
        if time_col is not None and time_col in group.columns:
            group = group.sort_values(time_col)
        sequence = [item_map[item] for item in group[item_col] if item in item_map]
        if sequence:
            user_sequences[reid] = sequence

    if not user_sequences:
        raise ValueError("No usable user sequences constructed from KuaiRec data.")

    print(
        f"Loaded KuaiRec interactions from {interaction_path} with "
        f"{len(user_sequences)} users and {len(item_map)} items.")

    return KuaiRecData(user_sequences=user_sequences, user_map=user_map, item_map=item_map)


class KuaiRecTrainDataset(Dataset):
    """Per-user sequence dataset for training."""

    def __init__(self, data: KuaiRecData, maxlen: int, min_history: int = 2):
        self.data = data
        self.maxlen = maxlen
        self.min_history = min_history
        self.user_ids = [
            user_id for user_id, seq in data.user_sequences.items() if len(seq) >= min_history
        ]
        self.item_ids = set(range(1, data.num_items + 1))

    def __len__(self) -> int:
        return len(self.user_ids)

    def _sample_negative(self, positives: Iterable[int]) -> int:
        positives_set = set(positives)
        if len(positives_set) >= self.data.num_items:
            # Degenerate case: fall back to padding token which will be ignored by the mask.
            return 0
        neg = np.random.randint(1, self.data.num_items + 1)
        while neg in positives_set:
            neg = np.random.randint(1, self.data.num_items + 1)
        return int(neg)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | int | Sequence[int]]:
        user_reid = self.user_ids[index]
        sequence = self.data.user_sequences[user_reid]
        history = sequence[:-1]
        targets = sequence[1:]

        seq_array = np.zeros(self.maxlen, dtype=np.int64)
        pos_array = np.zeros(self.maxlen, dtype=np.int64)
        neg_array = np.zeros(self.maxlen, dtype=np.int64)
        mask_array = np.zeros(self.maxlen, dtype=np.float32)

        history_items = set(sequence)
        pointer = self.maxlen - 1
        for item, target in zip(reversed(history), reversed(targets)):
            seq_array[pointer] = item
            pos_array[pointer] = target
            neg_array[pointer] = self._sample_negative(history_items)
            mask_array[pointer] = 1.0
            pointer -= 1
            if pointer < 0:
                break

        return {
            "user": user_reid,
            "seq": seq_array,
            "pos": pos_array,
            "neg": neg_array,
            "mask": mask_array,
            "history_items": list(history_items),
        }

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, object]]) -> Dict[str, torch.Tensor | List[List[int]]]:
        seq = torch.as_tensor(np.stack([sample["seq"] for sample in batch], axis=0), dtype=torch.long)
        pos = torch.as_tensor(np.stack([sample["pos"] for sample in batch], axis=0), dtype=torch.long)
        neg = torch.as_tensor(np.stack([sample["neg"] for sample in batch], axis=0), dtype=torch.long)
        mask = torch.as_tensor(np.stack([sample["mask"] for sample in batch], axis=0), dtype=torch.float32)
        user = torch.tensor([sample["user"] for sample in batch], dtype=torch.long)
        history_items = [sample["history_items"] for sample in batch]
        return {
            "seq": seq,
            "pos": pos,
            "neg": neg,
            "mask": mask,
            "user": user,
            "history_items": history_items,
        }


class KuaiRecEvalDataset(Dataset):
    """Dataset that holds out the last interaction for evaluation."""

    def __init__(self, data: KuaiRecData, maxlen: int):
        self.data = data
        self.maxlen = maxlen
        self.samples: List[Dict[str, object]] = []
        for user_reid, sequence in data.user_sequences.items():
            if len(sequence) < 2:
                continue
            history = sequence[:-1]
            target = sequence[-1]
            trimmed_history = history[-maxlen:]
            seq_array = np.zeros(maxlen, dtype=np.int64)
            seq_array[-len(trimmed_history) :] = trimmed_history
            self.samples.append(
                {
                    "user": user_reid,
                    "seq": seq_array,
                    "target": target,
                    "length": len(trimmed_history),
                    "history_items": set(trimmed_history),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.samples[index]

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        seq = torch.as_tensor(np.stack([sample["seq"] for sample in batch], axis=0), dtype=torch.long)
        target = torch.tensor([sample["target"] for sample in batch], dtype=torch.long)
        length = torch.tensor([sample["length"] for sample in batch], dtype=torch.long)
        user = torch.tensor([sample["user"] for sample in batch], dtype=torch.long)
        history_items = [sample["history_items"] for sample in batch]
        return {
            "seq": seq,
            "target": target,
            "length": length,
            "user": user,
            "history_items": history_items,
        }


__all__ = [
    "KuaiRecData",
    "KuaiRecTrainDataset",
    "KuaiRecEvalDataset",
    "load_kuairec_data",
    "compute_dataset_statistics",
]
