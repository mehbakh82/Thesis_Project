#!/usr/bin/env python3
"""Project-owned entry point for the pinned Moshi-Finetune train module."""

from __future__ import annotations

import json
import math
import os
import runpy
import sys
from pathlib import Path

from scripts.moshi_text_input_dropout import configure_scheduled_text_input_dropout

TEXT_EMBEDDING_PARAMETER_NAMES = frozenset(
    {
        "depformer_text_emb.weight",
        "text_emb.weight",
    }
)

PERSIAN_TEXT_PARAMETER_NAMES = frozenset(
    {
        "depformer_text_emb.weight",
        "text_emb.weight",
        "text_linear.frozen_W.weight",
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


def _configure_cpu_offloaded_adamw(
    torch_module, mixed_precision_module, audit_path: Path
) -> None:
    """Keep exact FP32 AdamW master/state tensors and updates on host memory."""

    original_adamw = torch_module.optim.AdamW

    class CPUOffloadedAdamW(original_adamw):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            requested = kwargs.get("foreach")
            if requested not in (None, False):
                raise RuntimeError("CPU-offloaded AdamW requires foreach=False")
            requested_fused = kwargs.get("fused")
            if requested_fused not in (None, False):
                raise RuntimeError("CPU-offloaded AdamW requires fused=False")
            kwargs["foreach"] = False
            kwargs["fused"] = False
            super().__init__(*args, **kwargs)
            self._cpu_offload_step_calls = 0

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            active_parameters = [
                parameter
                for group in self.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
            if not active_parameters:
                raise RuntimeError("CPU-offloaded AdamW observed no active parameters")
            master_devices = {parameter.data.device.type for parameter in active_parameters}
            master_dtypes = {str(parameter.data.dtype) for parameter in active_parameters}
            gradient_devices = {
                parameter.grad.device.type for parameter in active_parameters
            }
            gradient_dtypes = {str(parameter.grad.dtype) for parameter in active_parameters}
            moment_tensors = [
                state[name]
                for parameter in active_parameters
                for state in (self.state[parameter],)
                for name in ("exp_avg", "exp_avg_sq")
                if name in state
            ]
            moment_devices = {tensor.device.type for tensor in moment_tensors}
            moment_dtypes = {str(tensor.dtype) for tensor in moment_tensors}
            if (
                master_devices != {"cpu"}
                or master_dtypes != {"torch.float32"}
                or gradient_devices != {"cpu"}
                or gradient_dtypes != {"torch.float32"}
                or moment_devices != {"cpu"}
                or moment_dtypes != {"torch.float32"}
            ):
                raise RuntimeError("CPU-offloaded AdamW placement or dtype invariant failed")
            self._cpu_offload_step_calls += 1
            audit = {
                "schema_version": 1,
                "algorithm": "AdamW",
                "optimizer_step_calls": self._cpu_offload_step_calls,
                "active_parameter_tensors": len(active_parameters),
                "active_parameter_elements": sum(
                    parameter.numel() for parameter in active_parameters
                ),
                "master_devices": sorted(master_devices),
                "master_dtypes": sorted(master_dtypes),
                "gradient_devices": sorted(gradient_devices),
                "gradient_dtypes": sorted(gradient_dtypes),
                "moment_tensor_count": len(moment_tensors),
                "moment_devices": sorted(moment_devices),
                "moment_dtypes": sorted(moment_dtypes),
                "foreach": False,
                "fused": False,
            }
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            return result

    def prepare_mixed_precision(params, param_dtype, optim_dtype):
        if optim_dtype is not torch_module.float32:
            raise RuntimeError("CPU-offloaded AdamW requires FP32 optimizer tensors")
        with torch_module.no_grad():
            for parameter in params:
                if parameter.requires_grad:
                    parameter._mp_param = parameter.detach().to(  # type: ignore[attr-defined]
                        device="cpu", dtype=optim_dtype, copy=True
                    )
                parameter.data = parameter.data.to(param_dtype)

    def upcast_mixed_precision(params, optim_dtype):
        if optim_dtype is not torch_module.float32:
            raise RuntimeError("CPU-offloaded AdamW requires FP32 optimizer tensors")
        with torch_module.no_grad():
            for parameter in params:
                if parameter.requires_grad and parameter.grad is not None:
                    parameter._temp = parameter.data  # type: ignore[attr-defined]
                    parameter.data = parameter._mp_param  # type: ignore[attr-defined]
                    parameter.grad = parameter.grad.detach().to(
                        device="cpu", dtype=optim_dtype, copy=True
                    )

    def downcast_mixed_precision(params, param_dtype):
        with torch_module.no_grad():
            for parameter in params:
                if parameter.requires_grad and hasattr(parameter, "_temp"):
                    if parameter.data.device.type != "cpu":
                        raise RuntimeError("CPU-offloaded AdamW master tensor left host memory")
                    parameter._temp.copy_(parameter.data)  # type: ignore[attr-defined]
                    parameter.grad = None
                    parameter.data = parameter._temp  # type: ignore[attr-defined]
                    del parameter._temp  # type: ignore[attr-defined]

    torch_module.optim.AdamW = CPUOffloadedAdamW
    mixed_precision_module.prepare_mixed_precision = prepare_mixed_precision
    mixed_precision_module.upcast_mixed_precision = upcast_mixed_precision
    mixed_precision_module.downcast_mixed_precision = downcast_mixed_precision
    os.environ["MOSHI_ADAMW_FOREACH_EFFECTIVE"] = "false"
    os.environ["MOSHI_ADAMW_FUSED_EFFECTIVE"] = "false"
    os.environ["MOSHI_OPTIMIZER_CPU_OFFLOAD_EFFECTIVE"] = "true"
    os.environ["MOSHI_OPTIMIZER_DTYPE_EFFECTIVE"] = "float32"


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


def _configure_audio_loss_weight(loss_module, weight: float) -> None:
    """Apply an explicit global weight to the complete audio objective."""

    if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
        raise RuntimeError("MOSHI_AUDIO_LOSS_WEIGHT must be finite and in (0, 1]")
    original_compute_loss = loss_module.compute_loss_with_mask

    def compute_loss_with_mask(*args, **kwargs):
        mode = kwargs.get("mode")
        if mode is None and len(args) >= 4:
            mode = args[3]
        loss = original_compute_loss(*args, **kwargs)
        return loss * weight if mode == "audio" else loss

    loss_module.compute_loss_with_mask = compute_loss_with_mask
    os.environ["MOSHI_AUDIO_LOSS_WEIGHT_EFFECTIVE"] = str(weight)


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
                f"unexpected trainable embedding scope: {sorted(trainable_embeddings)}"
            )
        os.environ["MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE"] = ",".join(sorted(trainable_embeddings))
        print(
            "MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE="
            + os.environ["MOSHI_TEXT_EMBEDDINGS_ONLY_EFFECTIVE"],
            flush=True,
        )
        return model

    wrapped_model_module.get_fsdp_model = get_fsdp_model


def _configure_persian_text_adaptation(wrapped_model_module) -> None:
    """Train the untied text head and text embeddings while audio embeddings stay frozen."""

    original_get_fsdp_model = wrapped_model_module.get_fsdp_model

    def get_fsdp_model(args, checkpoint_info):
        if args.full_finetuning or not args.lora.enable or args.lora.ft_embed:
            raise RuntimeError(
                "Persian text adaptation requires LoRA, partial finetuning, and ft_embed=false"
            )
        model = original_get_fsdp_model(args, checkpoint_info)
        parameters = dict(model.named_parameters())
        missing = PERSIAN_TEXT_PARAMETER_NAMES - parameters.keys()
        if missing:
            raise RuntimeError(f"missing required Persian text parameters: {sorted(missing)}")
        for name in PERSIAN_TEXT_PARAMETER_NAMES:
            parameters[name].requires_grad = True
        trainable_full_parameters = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora" not in name
        }
        if trainable_full_parameters != PERSIAN_TEXT_PARAMETER_NAMES:
            raise RuntimeError(
                f"unexpected trainable full-parameter scope: {sorted(trainable_full_parameters)}"
            )
        trainable_audio_embeddings = {
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and (name.startswith("emb.") or name.startswith("depformer_emb."))
        }
        if trainable_audio_embeddings:
            raise RuntimeError(
                f"unexpected trainable audio embeddings: {sorted(trainable_audio_embeddings)}"
            )
        os.environ["MOSHI_PERSIAN_TEXT_ADAPTATION_EFFECTIVE"] = ",".join(
            sorted(trainable_full_parameters)
        )
        print(
            "MOSHI_PERSIAN_TEXT_ADAPTATION_EFFECTIVE="
            + os.environ["MOSHI_PERSIAN_TEXT_ADAPTATION_EFFECTIVE"],
            flush=True,
        )
        return model

    wrapped_model_module.get_fsdp_model = get_fsdp_model


def _configure_text_input_dropout_from_environment(
    root: Path,
    wrapped_model_module,
    torch_module,
    *,
    persian_text_policy: str,
) -> None:
    start_text = os.environ.get("MOSHI_TEXT_INPUT_DROPOUT_START", "0").strip()
    end_text = os.environ.get("MOSHI_TEXT_INPUT_DROPOUT_END", "0").strip()
    forwards_text = os.environ.get("MOSHI_TEXT_INPUT_DROPOUT_FORWARDS", "0").strip()
    seed_text = os.environ.get("MOSHI_TEXT_INPUT_DROPOUT_SEED", "0").strip()
    audit_text = os.environ.get("MOSHI_TEXT_INPUT_DROPOUT_AUDIT", "").strip()
    try:
        start_probability = float(start_text)
        end_probability = float(end_text)
        train_forwards = int(forwards_text)
        seed = int(seed_text)
    except ValueError as exc:
        raise RuntimeError("invalid scheduled text-input dropout environment") from exc

    if start_probability == 0.0 and end_probability == 0.0:
        if train_forwards != 0 or seed != 0 or audit_text:
            raise RuntimeError("disabled text-input dropout has unexpected controls")
        return
    if persian_text_policy != "1":
        raise RuntimeError("scheduled text-input dropout requires Persian-text adaptation")

    audit_path = (root / audit_text).resolve() if audit_text else None
    if audit_path is not None and root != audit_path and root not in audit_path.parents:
        raise RuntimeError("text-input dropout audit must stay inside the project")
    configure_scheduled_text_input_dropout(
        wrapped_model_module,
        torch_module,
        start_probability=start_probability,
        end_probability=end_probability,
        train_forwards=train_forwards,
        seed=seed,
        audit_path=audit_path,
    )


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
    import finetune.loss as finetune_loss
    import finetune.mixed_precision as finetune_mixed_precision
    import finetune.wrapped_model as finetune_wrapped_model
    import torch

    optimizer_cpu_offload = os.environ.get("MOSHI_OPTIMIZER_CPU_OFFLOAD", "0").strip()
    if optimizer_cpu_offload not in {"0", "1"}:
        raise RuntimeError("MOSHI_OPTIMIZER_CPU_OFFLOAD must be exactly 0 or 1")
    optimizer_audit_text = os.environ.get(
        "MOSHI_OPTIMIZER_CPU_OFFLOAD_AUDIT", ""
    ).strip()
    if optimizer_cpu_offload == "1":
        if not optimizer_audit_text:
            raise RuntimeError(
                "MOSHI_OPTIMIZER_CPU_OFFLOAD_AUDIT is required when CPU offload is enabled"
            )
        optimizer_audit_path = (root / optimizer_audit_text).resolve()
        if (
            root != optimizer_audit_path
            and root not in optimizer_audit_path.parents
        ):
            raise RuntimeError("optimizer CPU-offload audit must stay inside the project")
        _configure_cpu_offloaded_adamw(
            torch, finetune_mixed_precision, optimizer_audit_path
        )
    else:
        if optimizer_audit_text:
            raise RuntimeError("disabled optimizer CPU offload has an unexpected audit path")
        _configure_low_peak_adamw(torch)
    _configure_low_peak_checkpoints(finetune_checkpointing, finetune_distributed, torch)
    _configure_repeatable_eval_loader(finetune_data_loader)
    text_embedding_policy = os.environ.get("MOSHI_TEXT_EMBEDDINGS_ONLY", "0").strip()
    if text_embedding_policy not in {"0", "1"}:
        raise RuntimeError("MOSHI_TEXT_EMBEDDINGS_ONLY must be exactly 0 or 1")
    persian_text_policy = os.environ.get("MOSHI_PERSIAN_TEXT_ADAPTATION", "0").strip()
    if persian_text_policy not in {"0", "1"}:
        raise RuntimeError("MOSHI_PERSIAN_TEXT_ADAPTATION must be exactly 0 or 1")
    if text_embedding_policy == "1" and persian_text_policy == "1":
        raise RuntimeError("text-embedding-only and Persian-text modes are mutually exclusive")
    if text_embedding_policy == "1":
        _configure_text_embeddings_only(finetune_wrapped_model)
    elif persian_text_policy == "1":
        _configure_persian_text_adaptation(finetune_wrapped_model)
    _configure_text_input_dropout_from_environment(
        root,
        finetune_wrapped_model,
        torch,
        persian_text_policy=persian_text_policy,
    )
    audio_loss_weight_text = os.environ.get("MOSHI_AUDIO_LOSS_WEIGHT", "1.0").strip()
    try:
        audio_loss_weight = float(audio_loss_weight_text)
    except ValueError as exc:
        raise RuntimeError("MOSHI_AUDIO_LOSS_WEIGHT must be numeric") from exc
    _configure_audio_loss_weight(finetune_loss, audio_loss_weight)

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
