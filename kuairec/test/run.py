"""Convenience wrapper for running inference on the KuaiRec dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "kuairec" / "data"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "kuairec" / "eval_results"
INFER_ENTRYPOINT = Path(__file__).resolve().with_name("main.py")


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in sorted(path.iterdir()):
            if candidate.suffix == ".pt":
                return candidate
        raise FileNotFoundError(f"No .pt file found inside {path}")
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def build_command(args: argparse.Namespace, extra: Sequence[str]) -> Sequence[str]:
    cmd = [
        args.python,
        str(INFER_ENTRYPOINT),
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
        description="Run the baseline inference pipeline with KuaiRec defaults.",
    )
    parser.add_argument(
        "--dataset-root",
        type=_path,
        default=DEFAULT_DATA_ROOT,
        help="Folder that contains the KuaiRec CSV files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=_path,
        required=True,
        help="Path to the checkpoint directory or .pt file produced by training.",
    )
    parser.add_argument(
        "--result-dir",
        type=_path,
        default=DEFAULT_RESULT_DIR,
        help="Directory for inference artifacts (defaults to kuairec/eval_results).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device passed to test/infer.py (e.g. cpu or cuda).",
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
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
        help="Forward the --norm_first flag to the inference script.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch test/infer.py",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional arguments forwarded to test/infer.py. Add '--' before extra options, "
            "e.g. '-- --mm_emb_id 82'."
        ),
    )

    args = parser.parse_args(argv)

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"KuaiRec dataset not found at {dataset_root}. Place the CSV files there or pass --dataset-root."
        )

    result_dir = args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = _resolve_checkpoint(args.checkpoint)

    env = os.environ.copy()
    env["MODEL_OUTPUT_PATH"] = str(checkpoint)
    env["EVAL_DATA_PATH"] = str(dataset_root)
    env["EVAL_RESULT_PATH"] = str(result_dir)

    extra = args.extra_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = list(build_command(args, extra))

    print("[kuairec/test] Launching:")
    print(" ".join(cmd))
    if checkpoint != args.checkpoint:
        print(f"[kuairec/test] Resolved checkpoint: {checkpoint}")
    subprocess.run(cmd, check=True, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
