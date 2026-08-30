from __future__ import annotations

import pytest

from scripts.preflight_moshi_v4 import selective_adapter_bytes


def test_v4_adapter_estimate_combines_rank128_lora_and_only_text_embeddings() -> None:
    rank128 = [
        {"name": f"layer.{index}.lora_A.weight", "dtype": "BF16", "shape": [1, 2]}
        for index in range(674)
    ]
    broad = [
        {"name": "text_emb.weight", "dtype": "BF16", "shape": [10, 4]},
        {"name": "depformer_text_emb.weight", "dtype": "BF16", "shape": [10, 2]},
        {"name": "emb.0.weight", "dtype": "BF16", "shape": [100, 8]},
    ]

    estimate = selective_adapter_bytes(rank128, broad)

    assert estimate == {
        "lora_bytes": 2696,
        "text_embedding_bytes": 120,
        "estimated_adapter_bytes": 2816,
        "expected_tensor_count": 676,
    }


def test_v4_adapter_estimate_rejects_missing_text_embedding() -> None:
    rank128 = [
        {"name": f"layer.{index}.lora_A.weight", "dtype": "BF16", "shape": [1, 1]}
        for index in range(674)
    ]
    broad = [{"name": "text_emb.weight", "dtype": "BF16", "shape": [10, 4]}]

    with pytest.raises(ValueError, match="both exact text embeddings"):
        selective_adapter_bytes(rank128, broad)


def test_v4_adapter_estimate_rejects_non_lora_rank128_source() -> None:
    rank128 = [
        {"name": f"layer.{index}.lora_A.weight", "dtype": "BF16", "shape": [1, 1]}
        for index in range(674)
    ] + [{"name": "text_emb.weight", "dtype": "BF16", "shape": [1, 1]}]
    broad = [
        {"name": "text_emb.weight", "dtype": "BF16", "shape": [10, 4]},
        {"name": "depformer_text_emb.weight", "dtype": "BF16", "shape": [10, 2]},
    ]

    with pytest.raises(ValueError, match="no other tensors"):
        selective_adapter_bytes(rank128, broad)
