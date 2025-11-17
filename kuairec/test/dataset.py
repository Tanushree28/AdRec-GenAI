"""Evaluation-time dataset helpers for KuaiRec."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from kuairec.train.dataset import (
    KuaiRecData,
    compute_dataset_statistics,
    downsample_head_items,
    load_kuairec_data,
    trim_user_sequences,
)


def prepare_eval_data(
    dataset_root: Path,
    *,
    maxlen: int,
    trim_sequences_to: int | None = None,
    head_downsample_percent: float | None = None,
    head_downsample_keep_prob: float | None = None,
    head_downsample_seed: int | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Tuple[KuaiRecData, Dict[str, object], List[Dict[str, object]]]:
    """Load KuaiRec data and apply the same preprocessing used for training."""

    data = load_kuairec_data(dataset_root)
    transformations: List[Dict[str, object]] = []
    saved_args = metadata.get("args", {}) if metadata else {}

    trim_limit = (
        trim_sequences_to
        if trim_sequences_to is not None
        else saved_args.get("trim_sequences_to")
    )
    if not trim_limit or trim_limit <= 0:
        trim_limit = maxlen

    if trim_limit and trim_limit > 0:
        data, trim_meta = trim_user_sequences(data, int(trim_limit))
        trim_meta["type"] = "sequence_trim"
        if trim_meta.get("applied"):
            transformations.append(trim_meta)

    head_percent = head_downsample_percent
    if head_percent is None:
        head_percent = saved_args.get("head_downsample_percent")
    if head_percent is not None:
        head_percent = float(head_percent)

    keep_prob = head_downsample_keep_prob
    if keep_prob is None:
        keep_prob = saved_args.get("head_downsample_keep_prob")
    if keep_prob is not None:
        keep_prob = float(keep_prob)

    seed = head_downsample_seed
    if seed is None:
        seed = saved_args.get("head_downsample_seed", 42)
    if seed is not None:
        seed = int(seed)

    if head_percent and head_percent > 0 and keep_prob is not None and keep_prob < 1.0:
        data, down_meta = downsample_head_items(
            data,
            head_percent=head_percent,
            keep_probability=keep_prob,
            seed=seed,
        )
        down_meta["type"] = "head_downsampling"
        if down_meta.get("applied"):
            transformations.append(down_meta)

    dataset_stats = compute_dataset_statistics(data)
    dataset_stats["transformations"] = transformations
    return data, dataset_stats, transformations


__all__ = [
    "prepare_eval_data",
]
