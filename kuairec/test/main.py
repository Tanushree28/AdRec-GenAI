"""KuaiRec inference entrypoint that reuses the baseline infer script."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_SRC = PROJECT_ROOT / "test"

if str(TEST_SRC) not in sys.path:
    sys.path.insert(0, str(TEST_SRC))

from infer import main as infer_main  # noqa: E402


def main() -> int:
    os.environ.setdefault("EVAL_DATA_PATH", str(PROJECT_ROOT / "kuairec" / "data"))
    os.environ.setdefault("EVAL_RESULT_PATH", str(PROJECT_ROOT / "kuairec" / "eval_results"))
    return infer_main()


if __name__ == "__main__":
    raise SystemExit(main())
