#!/usr/bin/env python3
"""Project-owned entry point for the pinned Moshi-Finetune train module."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

TEXT_EMBEDDING_PARAMETER_NAMES = frozenset(
    {
        "depformer_text_emb.weight",
        "text_emb.weight",
    }
)


def _configure_low_peak_adamw(torch_module) -> None:
    """Use fused AdamW updates to avoid full-size CUDA denominator temporaries."""

    original_adamw = torch_module.optim.AdamW

    class LowPeakAdamW(original_adamw):  # type: ignore[misc, valid-type]
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


def _configure_low_peak_checkpoints(checkpointing_module, distributed_module, torch_module) -> None:
    """Offload single-GPU adapter checkpoint copies directly to host memory."""

    checkpointer = checkpointing_module.Checkpointer
    original_retrieve = checkpointer.retrieve_save_states

    def retrieve_save_states(self, save_only_lora, save_dtype):
        if distributed_module.get_world_size() != 1 or not save_only_lora:
            return original_retrieve(self, save_only_lora, save_dtype)
        if self.full_finetuning:
            raise AssertionError("Cannot save LoRA checkpoint as LoRA training is not enabled.")

        with torch_module.no_grad():
            for module in self.model.modules():
                if isinstance(module, checkpointing_module.LoRALinear) and hasattr(
                    module, "_merge_lora_handle"
                ):
                    module._merge_lora_handle.remove()

            modules = {
                key: module
                for key, module in self.model.named_modules()
                if all(parameter.requires_grad for parameter in module.parameters())
            }
            states = {}
            for key, module in modules.items():
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                states.update(
                    {
                        f"{parent_prefix}.{state_key}": value.detach().to(
                            device="cpu", dtype=save_dtype, copy=True
                        )
                        for state_key, value in module.state_dict().items()
                    }
                )

        return dict(sorted(states.items()))

    checkpointer.retrieve_save_states = retrieve_save_states
    os.environ["MOSHI_CHECKPOINT_CPU_OFFLOAD_EFFECTIVE"] = "true"


class _RepeatableEvalLoader:
    def __init__(self, factory):
        self._factory = factory

    def __iter__(self):
        return iter(self._factory())


def _configure_repeatable_eval_loader(data_loader_module) -> None:
    """Recreate the pinned trainer's finite evaluation iterator on every use."""

    original_build = data_loader_module.build_data_loader

    def build_data_loader(*args, **kwargs):
        is_eval = kwargs.get("is_eval")
        if is_eval is None and len(args) >= 7:
            is_eval = args[6]
        if not is_eval:
            return original_build(*args, **kwargs)
        return _RepeatableEvalLoader(lambda: original_build(*args, **kwargs))

    data_loader_module.build_data_loader = build_data_loader
    os.environ["MOSHI_REPEATABLE_EVAL_LOADER_EFFECTIVE"] = "true"


def _configure_text_embeddings_only(wrapped_model_module) -> None:
    """Train the two text embeddings while keeping every audio embedding frozen."""

    original_get_fsdp_model = wrapped_model_module.get_fsdp_model

    def get_fsdp_model(args, checkpoint_info):
        if args.full_finetuning or not args.lora.enable or args.lora.ft_embed:
            raise RuntimeError(
                "text-embedding-only mode requires LoRA, partial finetuning, and ft_embed=false"
            )
        model = original_get_fsdp_model(args, checkpoint_info)
        parameters = dict(model.named_parameters())
        missing = TEXT_EMBEDDING_PARAMETER_NAMES - parameters.keys()
        if missing:
            raise RuntimeError(f"missing required text embeddings: {sorted(missing)}")
        for name in TEXT_EMBEDDING_PARAMETER_NAMES:
            parameters[name].requires_grad = True
        trainable_embeddings = {
            name
            for name, parameter in model.named_parameters()
            if "emb" in name and parameter.requires_grad
        }
        if trainable_embeddings != TEXT_EMBEDDING_PARAMETER_NAMES:
            raise RuntimeError(
                "unexpected trainable embedding scope: "
                f"{sorted(trainable_embeddings)}"
            )
        os.environ["MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE"] = ",".join(
            sorted(trainable_embeddings)
        )
        print(
            "MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE="
            + os.environ["MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE"],
            flush=True,
        )
        return model

    wrapped_model_module.get_fsdp_model = get_fsdp_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checkout = root / "third_party" / "checkouts" / "moshi-finetune"
    train_module = checkout / "train.py"
    if not train_module.is_file():
        raise RuntimeError(f"pinned Moshi-Finetune checkout is missing: {train_module}")
    sys.path.insert(0, str(checkout))

    import finetune.checkpointing as finetune_checkpointing
    import finetune.data.data_loader as finetune_data_loader
    import finetune.distributed as finetune_distributed
    import finetune.wrapped_model as finetune_wrapped_model
    import torch

    _configure_low_peak_adamw(torch)
    _configure_low_peak_checkpoints(finetune_checkpointing, finetune_distributed, torch)
    _configure_repeatable_eval_loader(finetune_data_loader)
    text_embedding_policy = os.environ.get("MOSHI_TEXT_EMBEDDINGS_ONLY", "0").strip()
    if text_embedding_policy not in {"0", "1"}:
        raise RuntimeError("MOSHI_TEXT_EMBEDDINGS_ONLY must be exactly 0 or 1")
    if text_embedding_policy == "1":
        _configure_text_embeddings_only(finetune_wrapped_model)

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
