#!/usr/bin/env python3
"""Validate the isolated Moshi trainer environment without loading model weights."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch
from finetune.args import TrainArgs
from moshi.models import loaders

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "torchaudio", "triton", "moshi", "finetune", "sphn")
CHECKOUTS = {
    "Moshi": PROJECT_ROOT / "third_party" / "checkouts" / "moshi",
    "Moshi-Finetune": PROJECT_ROOT / "third_party" / "checkouts" / "moshi-finetune",
}


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "hardware" / "moshi_environment.json",
    )
    args = parser.parse_args()
    lock = json.loads(
        (PROJECT_ROOT / "third_party" / "UPSTREAMS.lock.json").read_text(encoding="utf-8")
    )
    expected_revisions = {row["name"]: row["revision"] for row in lock["upstreams"]}
    checkouts = {
        name: {
            "expected": expected_revisions.get(name),
            "actual": (actual := git_head(path)),
            "matches": actual == expected_revisions.get(name),
        }
        for name, path in CHECKOUTS.items()
    }
    legacy = json.loads(
        (PROJECT_ROOT / "configs" / "moshika_7b_legacy.json").read_text(encoding="utf-8")
    )
    model_type = legacy.pop("model_type", None)
    configs = {}
    for filename in ("moshi_h100.yaml", "moshi_h100_smoke.yaml"):
        parsed = TrainArgs.load(str(PROJECT_ROOT / "configs" / filename), drop_extra_fields=False)
        configs[filename] = {
            "max_steps": parsed.max_steps,
            "duration_sec": parsed.duration_sec,
            "lora": dataclasses.asdict(parsed.lora),
            "param_dtype": parsed.param_dtype,
        }
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gates = {
        "running_in_isolated_venv": Path(sys.prefix).resolve()
        == (PROJECT_ROOT / ".venv-moshi").resolve(),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "architecture_matches_pinned_loader": model_type == "moshi"
        and legacy == loaders._lm_kwargs,
        "checkout_revisions_match": all(row["matches"] for row in checkouts.values()),
        "configs_parse": len(configs) == 2,
    }
    report = {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "environment": sys.prefix,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "packages": packages,
        "checkouts": checkouts,
        "configs": configs,
        "gates": gates,
        "valid": all(gates.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
