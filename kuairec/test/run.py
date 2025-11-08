import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "kuairec" / "data"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "kuairec" / "eval_results"
INFER_ENTRYPOINT = "kuairec.test.main"


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt file found inside {path}")
        return candidates[-1]
    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def build_command(args: argparse.Namespace, checkpoint: Path, extra: Sequence[str]) -> Sequence[str]:
    cmd = [
        args.python,
        "-m",
        INFER_ENTRYPOINT,
        "--dataset-root",
        str(args.dataset_root),
        "--checkpoint",
        str(checkpoint),
        "--result-dir",
        str(args.result_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--maxlen",
        str(args.maxlen),
        "--hidden-units",
        str(args.hidden_units),
        "--num-blocks",
        str(args.num_blocks),
        "--num-heads",
        str(args.num_heads),
        "--dropout-rate",
        str(args.dropout_rate),
        "--topk",
        str(args.topk),
    ]

    if args.norm_first:
        cmd.append("--norm-first")

    cmd.extend(extra)
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the KuaiRec inference pipeline with package defaults.",
    )
    parser.add_argument("--dataset-root", type=_path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=_path, required=True)
    parser.add_argument("--result-dir", type=_path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--maxlen", type=int, default=101)
    parser.add_argument("--hidden-units", dest="hidden_units", type=int, default=32)
    parser.add_argument("--num-blocks", dest="num_blocks", type=int, default=1)
    parser.add_argument("--num-heads", dest="num_heads", type=int, default=1)
    parser.add_argument("--dropout-rate", dest="dropout_rate", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional options forwarded to kuairec.test.main (prefix with '--').",
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
    metadata_path = checkpoint.parent / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
        saved_args = metadata.get("args", {})
        for key in ["batch_size", "maxlen", "hidden_units", "num_blocks", "num_heads", "dropout_rate"]:
            if key in saved_args:
                setattr(args, key, saved_args[key])
        if "norm_first" in saved_args:
            args.norm_first = bool(saved_args["norm_first"])

    extra = args.extra_args
    if extra and extra[0] == "--":
        extra = extra[1:]

    cmd = list(build_command(args, checkpoint, extra))

    print("[kuairec/test] Launching:")
    print(" ".join(cmd))
    if checkpoint != args.checkpoint:
        print(f"[kuairec/test] Resolved checkpoint: {checkpoint}")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
