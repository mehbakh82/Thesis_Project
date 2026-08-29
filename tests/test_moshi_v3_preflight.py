from __future__ import annotations

import pytest

from scripts.preflight_moshi_v3 import (
    experiment_invariants,
    rank_scaled_adapter_bytes,
    training_shape,
)


def test_rank_scaled_adapter_estimate_excludes_embeddings() -> None:
    tensors = [
        {"name": "layer.lora_A", "dtype": "BF16", "shape": [64, 10]},
        {"name": "layer.lora_B", "dtype": "BF16", "shape": [10, 64]},
        {"name": "text_emb.weight", "dtype": "BF16", "shape": [1000, 10]},
    ]
    assert rank_scaled_adapter_bytes(tensors, source_rank=64, target_rank=128) == 5120


def test_rank_scaled_adapter_estimate_fails_without_lora() -> None:
    with pytest.raises(ValueError, match="no LoRA tensors"):
        rank_scaled_adapter_bytes(
            [{"name": "text_emb.weight", "dtype": "BF16", "shape": [10, 10]}],
            source_rank=64,
            target_rank=128,
        )


def test_v3_comparison_profiles_separate_invariants_and_shape() -> None:
    base = {
        "data": {"train_data": "train", "eval_data": "val", "shuffle": True},
        "moshi_paths": {"moshi_path": "model"},
        "full_finetuning": False,
        "lora": {"enable": True, "rank": 64, "scaling": 2.0, "ft_embed": True},
        "first_codebook_weight_multiplier": 100.0,
        "text_padding_weight": 0.5,
        "duration_sec": 100,
        "batch_size": 1,
        "num_microbatches": 4,
        "gradient_checkpointing": True,
        "optim": {"lr": 2e-6},
        "seed": 7,
        "log_freq": 10,
        "eval_freq": 100,
        "do_eval": True,
        "do_ckpt": True,
        "ckpt_freq": 100,
        "save_adapters": True,
        "overwrite_run_dir": False,
    }
    changed = {**base, "lora": {**base["lora"], "ft_embed": False}}

    assert experiment_invariants(base) == experiment_invariants(changed)
    assert training_shape(base) != training_shape(changed)
