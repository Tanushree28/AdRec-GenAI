"""Utility for inspecting binary inference artifacts.

Reads the files produced by ``test/infer.py`` (``query.fbin``, ``embedding.fbin``,
``id.u64bin``, optional ``id100.u64bin`` and ``retrive_id2creative_id.json``)
and prints human-readable summaries.  The script can also materialise the first
few recommendation lists as JSON when the ANN stage produced ``id100.u64bin``.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _read_matrix(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as fh:
        header = fh.read(8)
        if len(header) != 8:
            raise ValueError(f"{path} does not contain the expected 8 byte header")
        num_points, num_dim = struct.unpack("II", header)
        data = np.fromfile(fh, dtype=dtype, count=num_points * num_dim)
    try:
        return data.reshape(num_points, num_dim)
    except ValueError as exc:  # pragma: no cover - defensive, should not happen
        raise ValueError(
            f"Could not reshape payload from {path} into ({num_points}, {num_dim})"
        ) from exc


def _read_ann_results(path: Path) -> np.ndarray:
    with path.open("rb") as fh:
        header = fh.read(8)
        if len(header) != 8:
            raise ValueError(f"{path} does not contain the expected 8 byte header")
        num_points, top_k = struct.unpack("II", header)
        payload = np.fromfile(fh, dtype=np.uint64, count=num_points * top_k)
    return payload.reshape(num_points, top_k)


def _preview_rows(rows: Sequence[Sequence[int]], limit: int) -> Sequence[Sequence[int]]:
    return [list(row) for row in rows[:limit]]


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect inference artifacts")
    parser.add_argument(
        "result_dir",
        type=Path,
        help="Directory that contains query.fbin, embedding.fbin, id.u64bin, etc.",
    )
    parser.add_argument(
        "--sample-users",
        type=int,
        default=5,
        help="How many users to display when ANN results are available (default: 5)",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Optional path to dump the preview recommendations as JSON.",
    )
    args = parser.parse_args(argv)

    result_dir: Path = args.result_dir.expanduser().resolve()
    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory {result_dir} does not exist")

    print(f"[dump] Reading inference artifacts from {result_dir}")

    query_path = result_dir / "query.fbin"
    item_emb_path = result_dir / "embedding.fbin"
    item_id_path = result_dir / "id.u64bin"
    ann_path = result_dir / "id100.u64bin"
    mapping_path = result_dir / "retrive_id2creative_id.json"

    if query_path.exists():
        query = _read_matrix(query_path, np.float32)
        print(f"[dump] query.fbin     -> shape={query.shape}")
    else:
        query = None
        print(f"[dump] query.fbin     -> NOT FOUND")

    if item_emb_path.exists():
        item_emb = _read_matrix(item_emb_path, np.float32)
        print(f"[dump] embedding.fbin -> shape={item_emb.shape}")
    else:
        item_emb = None
        print(f"[dump] embedding.fbin -> NOT FOUND")

    if item_id_path.exists():
        item_ids = _read_matrix(item_id_path, np.uint64).reshape(-1)
        print(f"[dump] id.u64bin      -> {item_ids.size} ids (first 5: {item_ids[:5].tolist()})")
    else:
        item_ids = None
        print(f"[dump] id.u64bin      -> NOT FOUND")

    mapping = {}
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as fh:
            mapping = {int(k): v for k, v in json.load(fh).items()}
        print(
            "[dump] retrive_id2creative_id.json ->",
            f"{len(mapping)} entries (sample: {list(mapping.items())[:5]})",
        )
    else:
        print("[dump] retrive_id2creative_id.json -> NOT FOUND")

    preview = []
    if ann_path.exists():
        ann = _read_ann_results(ann_path)
        print(
            "[dump] id100.u64bin   ->",
            f"shape={ann.shape} (top_k={ann.shape[1] if ann.ndim == 2 else 'n/a'})",
        )
        preview = _preview_rows(ann, args.sample_users)
        if mapping:
            decoded = [
                [mapping.get(int(recall_id), None) for recall_id in row]
                for row in preview
            ]
            print("[dump] First", len(decoded), "users (creative IDs):")
            for idx, row in enumerate(decoded, start=1):
                print(f"    user[{idx:02d}]: {row}")
        else:
            print("[dump] ANN preview (retrieval ids):")
            for idx, row in enumerate(preview, start=1):
                print(f"    user[{idx:02d}]: {row}")
    else:
        print("[dump] id100.u64bin   -> NOT FOUND (ANN skipped during inference)")

    if args.write_json and preview:
        payload = {
            "retrieval_ids": preview,
            "creative_ids": [
                [mapping.get(int(recall_id), None) for recall_id in row]
                for row in preview
            ]
            if mapping
            else None,
        }
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        with args.write_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[dump] Wrote preview JSON to {args.write_json}")


if __name__ == "__main__":
    main()
