#!/usr/bin/env python3
"""Run the frozen recorded-audio interruption proxy benchmark once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thesis_s2s.bargein.recorded_proxy import evaluate_recorded_proxy  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bargein_recorded_proxy.yaml"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = evaluate_recorded_proxy(
        config_path=(ROOT / args.config).resolve(),
        out_path=(ROOT / args.out).resolve() if args.out is not None else None,
        root=ROOT,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
