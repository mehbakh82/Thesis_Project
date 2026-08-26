from __future__ import annotations

import pytest
import torch

from scripts.evaluate_moshi_adapter import (
    cyclically_perturb_masked_targets,
    expected_chunk_count,
    paired_difference,
    summarize,
)


def test_summarize_reports_mean_and_confidence_interval() -> None:
    result = summarize([1.0, 2.0, 3.0])
    assert result["samples"] == 3
    assert result["mean"] == 2.0
    assert result["ci95_low"] < 2.0 < result["ci95_high"]


def test_paired_difference_preserves_direction_and_nonzero() -> None:
    result = paired_difference(
        [3.0, 5.0],
        [2.0, 4.0],
        definition="left minus right",
    )
    assert result["mean"] == 1.0
    assert result["nonzero"] is True
    assert result["definition"] == "left minus right"


def test_paired_difference_rejects_unpaired_series() -> None:
    with pytest.raises(ValueError, match="same nonzero length"):
        paired_difference([1.0], [], definition="invalid")


def test_expected_chunk_count_uses_full_manifest(tmp_path) -> None:
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(
        '{"duration": 9.0}\n{"duration": 20.0}\n{"duration": 20.1}\n',
        encoding="utf-8",
    )
    assert expected_chunk_count(manifest, 20.0) == (3, 4)


@pytest.mark.parametrize("values", [[], [1.0, float("nan")]])
def test_summarize_rejects_empty_or_nonfinite(values) -> None:
    with pytest.raises(ValueError, match="nonempty and finite"):
        summarize(values)


def test_cyclic_target_perturbation_changes_only_masked_tokens() -> None:
    target = torch.tensor([[0, 1, 2], [2, 0, 1]])
    mask = torch.tensor([[True, False, True], [False, True, False]])

    perturbed, changed = cyclically_perturb_masked_targets(target, mask, cardinality=3)

    assert changed == 3
    assert torch.equal(perturbed, torch.tensor([[1, 1, 0], [2, 1, 1]]))
    assert torch.equal(target, torch.tensor([[0, 1, 2], [2, 0, 1]]))


def test_cyclic_target_perturbation_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="mask is empty"):
        cyclically_perturb_masked_targets(
            torch.tensor([0, 1]),
            torch.tensor([False, False]),
            cardinality=2,
        )
