"""Utilities for loading KuaiRec data independent of the Tencent baseline."""

from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

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
    temporal_stats: Mapping[str, object] | None = None
    category_map: Mapping[str, str] | None = None

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


def entropy_from_counts(values: Iterable[int]) -> float:
    total = float(sum(values))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in values:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p, 2)
    return entropy


def gini_from_counts(values: Iterable[int]) -> float:
    filtered = [float(v) for v in values if v > 0]
    if not filtered:
        return 0.0
    filtered.sort()
    n = len(filtered)
    cumulative = 0.0
    weighted_sum = 0.0
    for idx, value in enumerate(filtered, start=1):
        cumulative += value
        weighted_sum += idx * value
    return max(0.0, min(1.0, (2 * weighted_sum) / (n * cumulative) - (n + 1) / n))


def _compute_head_distribution(item_counts: Counter[int], num_items: int, total: int) -> Dict[str, Dict[str, float]]:
    if total == 0 or num_items == 0:
        return {
            "top_1_percent": {"count": 0, "share": 0.0},
            "top_5_percent": {"count": 0, "share": 0.0},
            "top_10_percent": {"count": 0, "share": 0.0},
        }

    sorted_counts = item_counts.most_common()

    def _share(percent: float) -> Dict[str, float]:
        head_size = max(1, int(math.ceil(num_items * percent)))
        head_total = sum(count for _, count in sorted_counts[:head_size])
        return {
            "count": head_size,
            "share": head_total / total if total else 0.0,
        }

    return {
        "top_1_percent": _share(0.01),
        "top_5_percent": _share(0.05),
        "top_10_percent": _share(0.10),
    }


def _compute_tail_distribution(item_counts: Counter[int], total: int) -> Dict[str, Dict[str, float]]:
    distribution: Dict[str, Dict[str, float]] = {}
    for threshold in (10, 5, 2):
        bucket_counts = [count for count in item_counts.values() if count < threshold]
        interaction_share = (sum(bucket_counts) / total) if total else 0.0
        distribution[f"<{threshold}"] = {
            "item_count": len(bucket_counts),
            "interaction_share": interaction_share,
        }
    return distribution


def _load_category_map(data_path: Path) -> Mapping[str, str] | None:
    candidates = [
        "item_categories.csv",
        "item_category.csv",
        "item_meta.csv",
        "item_features.csv",
        "video_features.csv",
    ]
    for candidate in candidates:
        path = data_path / candidate
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        item_col = _find_column(df.columns, ["item_id", "video_id", "cid", "creative", "item"])
        cat_col = _find_column(df.columns, ["category", "cate", "topic", "tag", "class"])
        if item_col is None or cat_col is None:
            continue
        df = df.dropna(subset=[item_col, cat_col])
        mapping = {str(row[item_col]): str(row[cat_col]) for _, row in df.iterrows()}
        if mapping:
            return mapping
    return None


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

    head_distribution = _compute_head_distribution(item_counts, data.num_items, total_interactions)
    tail_distribution = _compute_tail_distribution(item_counts, total_interactions)
    long_tail_ratio = tail_distribution.get("<10", {}).get("interaction_share", 0.0)
    popularity_bias_score = min(
        1.0,
        0.5 * head_distribution["top_1_percent"]["share"]
        + 0.3 * head_distribution["top_5_percent"]["share"]
        + 0.2 * head_distribution["top_10_percent"]["share"],
    )

    dominant_items = []
    for item_idx, count in item_counts.items():
        coverage = count / data.num_users if data.num_users else 0.0
        if coverage >= 0.9:
            dominant_items.append(
                {
                    "item_id": item_inverse.get(item_idx, str(item_idx)),
                    "count": int(count),
                    "user_coverage": coverage,
                }
            )
    dominant_items.sort(key=lambda entry: entry["count"], reverse=True)

    unique_last100 = [
        len(set(seq[-100:]))
        for seq in data.user_sequences.values()
        if seq
    ]
    unique_stats = {
        "avg": float(mean(unique_last100)) if unique_last100 else 0.0,
        "median": float(median(unique_last100)) if unique_last100 else 0.0,
        "min": int(min(unique_last100)) if unique_last100 else 0,
        "max": int(max(unique_last100)) if unique_last100 else 0,
    }

    category_diversity = None
    if data.category_map:
        inverse = data.item_inverse
        category_lookup = {
            idx: data.category_map.get(inverse.get(idx, ""))
            for idx in inverse.keys()
        }
        category_counts = []
        for seq in data.user_sequences.values():
            categories = [
                category_lookup.get(item)
                for item in seq[-100:]
                if category_lookup.get(item)
            ]
            if categories:
                category_counts.append(len(set(categories)))
        if category_counts:
            category_diversity = {
                "avg": float(mean(category_counts)),
                "median": float(median(category_counts)),
                "min": int(min(category_counts)),
                "max": int(max(category_counts)),
            }

    temporal_stats = data.temporal_stats or {}
    recent_ratio = temporal_stats.get("recent_30d_ratio")

    item_entropy = entropy_from_counts(item_counts.values())
    item_gini = gini_from_counts(item_counts.values())

    top5_total = sum(count for _, count in item_counts.most_common(5))
    top_item_full_coverage = any(item["count"] >= data.num_users for item in top_items)

    avg_per_user = total_interactions / data.num_users if data.num_users else 0.0
    avg_per_item = total_interactions / data.num_items if data.num_items else 0.0
    top5_share = top5_total / total_interactions if total_interactions else 0.0

    personalization_metrics = {
        "unique_items_last_100": unique_stats,
        "category_diversity_last_100": category_diversity,
        "recent_30d_interaction_ratio": recent_ratio,
        "item_entropy_bits": item_entropy,
        "item_gini": item_gini,
    }

    return {
        "num_users": data.num_users,
        "num_items": data.num_items,
        "total_interactions": total_interactions,
        "avg_sequence_length": avg_len,
        "median_sequence_length": median_len,
        "min_sequence_length": min_len,
        "max_sequence_length": max_len,
        "top_items": top_items,
        "avg_interactions_per_user": avg_per_user,
        "avg_interactions_per_item": avg_per_item,
        "top5_interaction_share": top5_share,
        "top_item_hits_all_users": top_item_full_coverage,
        "head_item_distribution": head_distribution,
        "tail_item_distribution": tail_distribution,
        "long_tail_ratio": long_tail_ratio,
        "popularity_bias_score": popularity_bias_score,
        "dominant_items": dominant_items,
        "item_entropy_bits": item_entropy,
        "item_gini": item_gini,
        "personalization_metrics": personalization_metrics,
        "temporal_stats": temporal_stats,
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

    # KUAIREC_MATRIX=big forces big_matrix.csv even when small_matrix.csv exists.
    preferred = os.environ.get("KUAIREC_MATRIX", "small").strip().lower()
    matrix_order = (
        ("big_matrix.csv", "small_matrix.csv")
        if preferred == "big"
        else ("small_matrix.csv", "big_matrix.csv")
    )
    interaction_path = data_path / matrix_order[0]
    if not interaction_path.exists():
        interaction_path = data_path / matrix_order[1]
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

    temporal_stats: Dict[str, object] | None = None
    if time_col is not None and time_col in interactions.columns:
        interactions = interactions.sort_values(time_col)
        timestamps = pd.to_datetime(interactions[time_col], errors="coerce")
        valid_ts = timestamps.dropna()
        if not valid_ts.empty:
            max_ts = valid_ts.max()
            min_ts = valid_ts.min()
            recent_threshold = max_ts - pd.Timedelta(days=30)
            recent_ratio = float((valid_ts >= recent_threshold).sum() / len(valid_ts))
            span_days = (max_ts - min_ts).total_seconds() / 86400 if max_ts != min_ts else 0.0
            temporal_stats = {
                "min_timestamp": min_ts.isoformat(),
                "max_timestamp": max_ts.isoformat(),
                "recent_30d_ratio": recent_ratio,
                "timespan_days": span_days,
            }

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

    category_map = _load_category_map(data_path)

    return KuaiRecData(
        user_sequences=user_sequences,
        user_map=user_map,
        item_map=item_map,
        temporal_stats=temporal_stats,
        category_map=category_map,
    )


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


def trim_user_sequences(
    data: "KuaiRecData", max_sequence_length: int
) -> Tuple["KuaiRecData", Dict[str, object]]:
    """Trim each user history to the most recent ``max_sequence_length`` items."""

    metadata: Dict[str, object] = {
        "applied": False,
        "max_sequence_length": max_sequence_length,
        "removed_interactions": 0,
        "affected_users": 0,
    }

    if max_sequence_length <= 0:
        return data, metadata

    trimmed_sequences: Dict[int, List[int]] = {}
    removed_interactions = 0
    affected_users = 0

    for user_id, sequence in data.user_sequences.items():
        if len(sequence) > max_sequence_length:
            affected_users += 1
            removed_interactions += len(sequence) - max_sequence_length
            trimmed_sequences[user_id] = list(sequence[-max_sequence_length:])
        else:
            trimmed_sequences[user_id] = list(sequence)

    if affected_users == 0:
        return data, metadata

    metadata["applied"] = True
    metadata["affected_users"] = affected_users
    metadata["removed_interactions"] = removed_interactions

    trimmed_data = KuaiRecData(
        user_sequences=trimmed_sequences,
        user_map=data.user_map,
        item_map=data.item_map,
        temporal_stats=data.temporal_stats,
        category_map=data.category_map,
    )
    return trimmed_data, metadata


def downsample_head_items(
    data: "KuaiRecData",
    head_percent: float = 0.01,
    keep_probability: float = 0.3,
    seed: int | None = None,
    min_history: int = 2,
) -> Tuple["KuaiRecData", Dict[str, object]]:
    """Randomly drop interactions from the head of the popularity distribution."""

    metadata: Dict[str, object] = {
        "applied": False,
        "head_percent": head_percent,
        "keep_probability": keep_probability,
        "removed_interactions": 0,
        "affected_items": 0,
        "affected_users": 0,
    }

    if keep_probability >= 1.0 or head_percent <= 0.0 or data.num_items == 0:
        return data, metadata

    item_counts: Counter[int] = Counter()
    for sequence in data.user_sequences.values():
        item_counts.update(sequence)

    head_size = max(1, int(math.ceil(data.num_items * head_percent)))
    head_items = {item for item, _ in item_counts.most_common(head_size)}
    if not head_items:
        return data, metadata

    rng = np.random.default_rng(seed)
    new_sequences: Dict[int, List[int]] = {}
    removed_interactions = 0
    affected_users = 0

    for user_id, sequence in data.user_sequences.items():
        filtered: List[int] = []
        skipped = 0
        for item in sequence:
            if item in head_items and rng.random() > keep_probability:
                skipped += 1
                continue
            filtered.append(item)

        restored = 0
        if len(filtered) < min_history and len(sequence) >= min_history:
            restored_slice = list(sequence[-min_history:])
            restored = len(restored_slice) - len(filtered)
            filtered = restored_slice

        removed_interactions += max(0, skipped - restored)
        if filtered:
            new_sequences[user_id] = filtered
            if skipped > 0:
                affected_users += 1

    if removed_interactions == 0:
        return data, metadata

    metadata["applied"] = True
    metadata["head_items_considered"] = len(head_items)
    metadata["affected_items"] = len(head_items)
    metadata["removed_interactions"] = removed_interactions
    metadata["affected_users"] = affected_users

    downsampled_data = KuaiRecData(
        user_sequences=new_sequences,
        user_map=data.user_map,
        item_map=data.item_map,
        temporal_stats=data.temporal_stats,
        category_map=data.category_map,
    )
    return downsampled_data, metadata


__all__ = [
    "KuaiRecData",
    "KuaiRecTrainDataset",
    "KuaiRecEvalDataset",
    "load_kuairec_data",
    "compute_dataset_statistics",
    "trim_user_sequences",
    "downsample_head_items",
    "entropy_from_counts",
    "gini_from_counts",
]
