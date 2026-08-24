#!/usr/bin/env python3
"""Select the Moshi adapter by the predeclared validation-loss rule."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from thesis_s2s.config import load_yaml, portable_project_values
from thesis_s2s.metrics import write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_checkpoint(run_dir: Path, config_path: Path, out_path: Path) -> dict:
    config = load_yaml(config_path)
    max_steps = int(config["max_steps"])
    checkpoint_frequency = int(config["ckpt_freq"])
    metrics_path = run_dir / "metrics.eval.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    candidates = []
    for row in rows:
        step = int(row.get("step") or 0)
        loss = float(row.get("eval_loss"))
        if (
            step <= 0
            or step > max_steps
            or step % checkpoint_frequency != 0
            or not math.isfinite(loss)
        ):
            continue
        consolidated = run_dir / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"
        adapter = consolidated / "lora.safetensors"
        adapter_config = consolidated / "config.json"
        if not adapter.is_file() or not adapter_config.is_file():
            continue
        candidates.append(
            {
                "step": step,
                "eval_loss": loss,
                "text_eval_loss": row.get("text_eval_loss"),
                "audio_eval_loss": row.get("audio_eval_loss"),
                "adapter_path": str(adapter),
                "adapter_sha256": sha256_file(adapter),
                "adapter_bytes": adapter.stat().st_size,
                "config_path": str(adapter_config),
                "config_sha256": sha256_file(adapter_config),
            }
        )
    if not rows or max(int(row.get("step") or 0) for row in rows) != max_steps:
        raise RuntimeError("training/evaluation log is incomplete; final configured step is absent")
    expected_candidates = max_steps // checkpoint_frequency
    if len(candidates) != expected_candidates:
        raise RuntimeError(
            f"expected {expected_candidates} finite checkpoint-aligned candidates, "
            f"found {len(candidates)}"
        )
    selected = min(candidates, key=lambda row: (float(row["eval_loss"]), int(row["step"])))
    report = {
        "schema_version": 1,
        "criterion_predeclared_before_training": True,
        "criterion": (
            "minimum finite mean validation eval_loss among every 500-step saved "
            "checkpoint; exact ties choose the earlier step"
        ),
        "heldout_test_used_for_selection": False,
        "training_complete": True,
        "configured_max_steps": max_steps,
        "checkpoint_frequency": checkpoint_frequency,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "selection_passes": True,
        "next_stage": (
            "load the selected adapter in the pinned runtime, then evaluate the "
            "group-disjoint test split once; never revise this choice from test or "
            "subjective outputs"
        ),
    }
    report = portable_project_values(report)
    write_json(out_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("checkpoints/moshi_fa"))
    parser.add_argument("--config", type=Path, default=Path("configs/moshi_h100.yaml"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/moshi_checkpoint_selection.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            select_checkpoint(args.run_dir, args.config, args.out),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
