from __future__ import annotations

import pytest

from scripts.validate_moshi_adapter import (
    compare_adapter_schema,
    normalise_lm_config,
    project_path,
    selected_candidate_by_rule,
)


def test_compare_adapter_schema_requires_exact_keys_shapes_and_dtypes() -> None:
    expected = {
        "text_emb.weight": {"shape": [4, 2], "dtype": "BF16"},
        "layer.lora_A.weight": {"shape": [1, 2], "dtype": "BF16"},
    }
    exact = compare_adapter_schema(expected, dict(expected))
    assert exact["exact"] is True

    mismatch = compare_adapter_schema(
        expected,
        {
            "text_emb.weight": {"shape": [4, 3], "dtype": "BF16"},
            "layer.lora_B.weight": {"shape": [2, 1], "dtype": "F16"},
        },
    )
    assert mismatch["exact"] is False
    assert mismatch["missing_keys"] == ["layer.lora_A.weight"]
    assert mismatch["unexpected_keys"] == ["layer.lora_B.weight"]
    assert mismatch["shape_mismatches"] == ["text_emb.weight"]


def test_normalise_lm_config_removes_only_loader_metadata() -> None:
    config = normalise_lm_config(
        {
            "model_type": "moshi",
            "lm_gen_config": {"temp": 0.8},
            "lora": True,
            "lora_rank": 64,
            "dim": 4096,
        }
    )
    assert config == {
        "lora": True,
        "lora_rank": 64,
        "dim": 4096,
    }


def test_selected_candidate_rule_uses_loss_then_earlier_step() -> None:
    candidates = [
        {"step": 500, "eval_loss": 1.5},
        {"step": 1000, "eval_loss": 1.0},
        {"step": 1500, "eval_loss": 1.0},
    ]
    assert selected_candidate_by_rule(candidates) == candidates[1]


@pytest.mark.parametrize(
    "candidate",
    [
        {"step": 0, "eval_loss": 1.0},
        {"step": 500, "eval_loss": float("nan")},
    ],
)
def test_selected_candidate_rule_rejects_invalid_candidate(candidate) -> None:
    with pytest.raises(ValueError, match="invalid step/eval_loss"):
        selected_candidate_by_rule([candidate])


def test_project_path_rejects_escape_and_absolute_paths(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    artifact = root / "adapter.safetensors"
    artifact.write_bytes(b"adapter")

    assert project_path(root, "adapter.safetensors", "adapter") == artifact
    with pytest.raises(ValueError, match="project-relative"):
        project_path(root, str(artifact), "adapter")
    with pytest.raises(ValueError, match="escapes"):
        project_path(root, "../outside.safetensors", "adapter")
    link = root / "adapter-link.safetensors"
    link.symlink_to(artifact)
    with pytest.raises(ValueError, match="not a regular file"):
        project_path(root, "adapter-link.safetensors", "adapter")
