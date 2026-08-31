from __future__ import annotations

import pytest

from scripts.preflight_moshi_v5 import selective_adapter_bytes


def test_v5_adapter_estimate_combines_rank128_lora_and_only_text_embeddings() -> None:
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


def test_v5_adapter_estimate_rejects_missing_text_embedding() -> None:
    rank128 = [
        {"name": f"layer.{index}.lora_A.weight", "dtype": "BF16", "shape": [1, 1]}
        for index in range(674)
    ]
    broad = [{"name": "text_emb.weight", "dtype": "BF16", "shape": [10, 4]}]

    with pytest.raises(ValueError, match="both exact text embeddings"):
        selective_adapter_bytes(rank128, broad)


def test_v5_adapter_estimate_rejects_non_lora_rank128_source() -> None:
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


def test_v5_changes_only_the_predeclared_objective_weight() -> None:
    import json
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    v4 = yaml.safe_load((root / "configs/moshi_h100_v4.yaml").read_text())
    v5 = yaml.safe_load((root / "configs/moshi_h100_v5.yaml").read_text())
    policy = json.loads((root / "configs/moshi_v5_embedding_policy.json").read_text())
    ignored = {"first_codebook_weight_multiplier", "run_dir"}
    assert {key: value for key, value in v4.items() if key not in ignored} == {
        key: value for key, value in v5.items() if key not in ignored
    }
    assert v4["first_codebook_weight_multiplier"] == 100.0
    assert v5["first_codebook_weight_multiplier"] == 10.0
    assert policy["objective_change"]["changed_parameter"] == ("first_codebook_weight_multiplier")
