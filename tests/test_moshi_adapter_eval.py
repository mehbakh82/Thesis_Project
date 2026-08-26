from __future__ import annotations

import pytest

from scripts.evaluate_moshi_adapter import (
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
