"""Scheduled text-input corruption for the train-only Moshi exposure-bias diagnostic."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

TEXT_INPUT_PRESERVED_MAX_ID = 3


def configure_scheduled_text_input_dropout(
    wrapped_model_module: Any,
    torch_module: Any,
    *,
    start_probability: float,
    end_probability: float,
    train_forwards: int,
    seed: int,
    audit_path: Path | None = None,
) -> None:
    """Corrupt lexical text inputs during training while retaining clean targets and eval."""

    if not all(math.isfinite(value) for value in (start_probability, end_probability)):
        raise RuntimeError("text-input dropout probabilities must be finite")
    if not 0.0 <= start_probability <= end_probability < 1.0:
        raise RuntimeError("text-input dropout requires 0 <= start <= end < 1")
    if end_probability == 0.0:
        raise RuntimeError("text-input dropout must have a positive end probability")
    if train_forwards <= 0:
        raise RuntimeError("text-input dropout train_forwards must be positive")
    if seed < 0:
        raise RuntimeError("text-input dropout seed must be nonnegative")
    if audit_path is not None and audit_path.exists():
        raise FileExistsError(f"refusing to overwrite text-input dropout audit: {audit_path}")

    original_get_fsdp_model = wrapped_model_module.get_fsdp_model

    def get_fsdp_model(args: Any, checkpoint_info: Any) -> Any:
        model = original_get_fsdp_model(args, checkpoint_info)
        if int(model.text_padding_token_id) != TEXT_INPUT_PRESERVED_MAX_ID:
            raise RuntimeError("scheduled text-input dropout requires text padding id 3")
        original_forward = model.forward
        generator = None
        train_forward_calls = 0
        cumulative_eligible = 0
        cumulative_dropped = 0
        milestones = {
            1,
            max(1, train_forwards // 4),
            max(1, train_forwards // 2),
            max(1, 3 * train_forwards // 4),
            train_forwards,
        }

        def forward(*args: Any, **kwargs: Any) -> Any:
            nonlocal generator, train_forward_calls, cumulative_eligible, cumulative_dropped
            codes = kwargs.get("codes")
            codes_position = None
            if codes is None and args:
                codes = args[0]
                codes_position = 0
            if not model.training or codes is None:
                return original_forward(*args, **kwargs)
            if codes.ndim != 3 or codes.shape[1] != model.num_codebooks:
                raise RuntimeError("unexpected codes shape for scheduled text-input dropout")
            if train_forward_calls >= train_forwards:
                raise RuntimeError("scheduled text-input dropout exceeded frozen train forwards")
            if generator is None:
                generator = torch_module.Generator(device=codes.device)
                generator.manual_seed(seed)

            progress = train_forward_calls / max(train_forwards - 1, 1)
            probability = start_probability + (end_probability - start_probability) * progress
            text_codes = codes[:, 0]
            eligible = text_codes > TEXT_INPUT_PRESERVED_MAX_ID
            random_values = torch_module.rand(
                text_codes.shape,
                device=text_codes.device,
                generator=generator,
            )
            dropped = eligible & (random_values < probability)
            input_codes = codes.clone()
            input_codes[:, 0].masked_fill_(dropped, int(model.text_padding_token_id))

            train_forward_calls += 1
            eligible_count = int(eligible.sum().item())
            dropped_count = int(dropped.sum().item())
            cumulative_eligible += eligible_count
            cumulative_dropped += dropped_count
            model.moshi_text_input_dropout_state = {
                "train_forward_calls": train_forward_calls,
                "probability": probability,
                "eligible_tokens": cumulative_eligible,
                "dropped_tokens": cumulative_dropped,
            }
            if audit_path is not None and train_forward_calls in milestones:
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "train_forward_call": train_forward_calls,
                    "probability": probability,
                    "eligible_tokens_this_call": eligible_count,
                    "dropped_tokens_this_call": dropped_count,
                    "cumulative_eligible_tokens": cumulative_eligible,
                    "cumulative_dropped_tokens": cumulative_dropped,
                    "seed": seed,
                    "preserved_token_ids_at_most": TEXT_INPUT_PRESERVED_MAX_ID,
                    "targets_mutated": False,
                    "evaluation_corrupted": False,
                }
                with audit_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")

            if codes_position is not None:
                forwarded_args = list(args)
                forwarded_args[codes_position] = input_codes
                return original_forward(*forwarded_args, **kwargs)
            forwarded_kwargs = dict(kwargs)
            forwarded_kwargs["codes"] = input_codes
            return original_forward(*args, **forwarded_kwargs)

        model.forward = forward
        os.environ["MOSHI_TEXT_INPUT_DROPOUT_EFFECTIVE"] = (
            f"{start_probability}:{end_probability}:{train_forwards}:{seed}"
        )
        print(
            "MOSHI_TEXT_INPUT_DROPOUT_EFFECTIVE="
            + os.environ["MOSHI_TEXT_INPUT_DROPOUT_EFFECTIVE"],
            flush=True,
        )
        return model

    wrapped_model_module.get_fsdp_model = get_fsdp_model
