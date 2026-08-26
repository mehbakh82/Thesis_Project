#!/usr/bin/env python3
"""Project-owned entry point for the pinned Moshi-Finetune train module."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def _configure_low_peak_adamw(torch_module) -> None:
    """Use fused AdamW updates to avoid full-size CUDA denominator temporaries."""

    original_adamw = torch_module.optim.AdamW

    class LowPeakAdamW(original_adamw):
        def __init__(self, *args, **kwargs):
            requested = kwargs.get("foreach")
            if requested not in (None, False):
                raise RuntimeError("the shared-H100 launcher requires AdamW foreach=False")
            requested_fused = kwargs.get("fused")
            if requested_fused not in (None, True):
                raise RuntimeError("the shared-H100 launcher requires AdamW fused=True")
            kwargs["foreach"] = False
            kwargs["fused"] = True
            super().__init__(*args, **kwargs)

    torch_module.optim.AdamW = LowPeakAdamW
    os.environ["MOSHI_ADAMW_FOREACH_EFFECTIVE"] = "false"
    os.environ["MOSHI_ADAMW_FUSED_EFFECTIVE"] = "true"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checkout = root / "third_party" / "checkouts" / "moshi-finetune"
    train_module = checkout / "train.py"
    if not train_module.is_file():
        raise RuntimeError(f"pinned Moshi-Finetune checkout is missing: {train_module}")
    sys.path.insert(0, str(checkout))

    import finetune.distributed as finetune_distributed
    import torch

    _configure_low_peak_adamw(torch)

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                "set CUDA_VISIBLE_DEVICES explicitly when more than one GPU is visible"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    backend = os.environ.get("MOSHI_DISTRIBUTED_BACKEND", "nccl").strip().lower()
    if backend not in {"nccl", "gloo"}:
        raise RuntimeError("MOSHI_DISTRIBUTED_BACKEND must be nccl or gloo")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if backend == "gloo" and world_size != 1:
        raise RuntimeError("the Gloo compatibility fallback is restricted to one GPU")
    finetune_distributed.BACKEND = backend
    runpy.run_module("train", run_name="__main__")


if __name__ == "__main__":
    main()
