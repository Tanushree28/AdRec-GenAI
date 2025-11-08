"""Convenience wrapper for training on the KuaiRec dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "kuairec" / "data"
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "kuairec"
DEFAULT_EVENTS_DIR = PROJECT_ROOT / "events" / "kuairec"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "kuairec" / "ckpt"
TRAIN_ENTRYPOINT = Path(__file__).resolve().with_name("main.py")


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_command(args: argparse.Namespace, extra: Sequence[str]) -> Sequence[str]:
    cmd = [
        args.python,
        str(TRAIN_ENTRYPOINT),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--maxlen",
        str(args.maxlen),
        "--hidden_units",
        str(args.hidden_units),
        "--num_blocks",
        str(args.num_blocks),
        "--num_epochs",
        str(args.num_epochs),
        "--num_heads",
        str(args.num_heads),
        "--dropout_rate",
        str(args.dropout_rate),
        "--l2_emb",
        str(args.l2_emb),
        "--device",
        args.device,
    ]

    if args.norm_first:
        cmd.append("--norm_first")

    cmd.extend(extra)
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the baseline training loop with KuaiRec defaults.",
    )
    parser.add_argument(
        "--dataset-root",
        type=_path,
        default=DEFAULT_DATA_ROOT,
        help="Folder that contains the KuaiRec CSV files (small_matrix.csv, user_features.csv, etc.)",
    )
    parser.add_argument(
        "--log-dir",
        type=_path,
        default=DEFAULT_LOG_DIR,
        help="Directory for JSON logs (defaults to logs/kuairec).",
    )
    parser.add_argument(
        "--events-dir",
        type=_path,
        default=DEFAULT_EVENTS_DIR,
        help="Directory for TensorBoard event files (defaults to events/kuairec).",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=_path,
        default=DEFAULT_CKPT_DIR,
        help="Directory where checkpoints should be stored (defaults to kuairec/ckpt).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device passed to train/main.py (e.g. cpu or cuda).",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--maxlen", type=int, default=101)
    parser.add_argument("--hidden-units", dest="hidden_units", type=int, default=32)
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=1)
    parser.add_argument("--num-epochs", dest="num_epochs", type=int, default=3)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=1)
    parser.add_argument("--dropout-rate", dest="dropout_rate", type=float, default=0.2)
    parser.add_argument("--l2-emb", dest="l2_emb", type=float, default=0.0)
    parser.add_argument(
        "--norm-first",
        action="store_true",
        help="Forward the --norm_first flag to the training script.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch train/main.py",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help=(
            "Additional arguments forwarded to train/main.py. Add '--' before the extra options, "
            "e.g. '-- --mm_emb_id 82'."
        ),
        default=[],
    )

    args = parser.parse_args(argv)

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"KuaiRec dataset not found at {dataset_root}. Place the CSV files there or pass --dataset-root."
        )

    for path in [args.log_dir, args.events_dir, args.ckpt_dir]:
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TRAIN_DATA_PATH"] = str(dataset_root)
    env["TRAIN_LOG_PATH"] = str(args.log_dir)
    env["TRAIN_TF_EVENTS_PATH"] = str(args.events_dir)
    env["TRAIN_CKPT_PATH"] = str(args.ckpt_dir)

    extra = args.extra_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = list(build_command(args, extra))

    print("[kuairec/train] Launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
