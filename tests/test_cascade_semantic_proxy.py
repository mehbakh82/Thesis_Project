from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_cascade_semantic_proxy import (
    EXPECTED_JUDGE_CALLS,
    JUDGE_IMAGE_ID,
    JUDGE_MODEL_ROOT,
    JUDGE_REVISION,
    JUDGE_SERVED_MODEL,
    MANIFEST_SHA256,
    PANEL_INDICES,
    PARENT_PANEL_SHA256,
    SCORE_KEYS,
    SYSTEM_PROMPT,
    parse_score_content,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_score_parser_is_fail_closed() -> None:
    assert parse_score_content('{"relevance": 4, "coherence": 3}') == {
        "relevance": 4,
        "coherence": 3,
    }
    for invalid in (
        '{"relevance": 5, "coherence": 3}',
        '{"relevance": true, "coherence": 3}',
        '{"relevance": 3.0, "coherence": 3}',
        '{"relevance": 3}',
        '{"relevance": 3, "coherence": 3, "rationale": "x"}',
        "not json",
    ):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_score_content(invalid)


def test_summary_reports_all_calls_repeats_and_fixed_dimensions() -> None:
    samples = [
        {"scores": [{"relevance": 4, "coherence": 3}] * 2},
        {
            "scores": [
                {"relevance": 2, "coherence": 2},
                {"relevance": 3, "coherence": 2},
            ]
        },
    ]
    report = summarize(samples)
    assert EXPECTED_JUDGE_CALLS == 18
    assert report["rows"] == 2
    assert report["judge_calls_valid"] == 4
    assert report["judge_calls_failed"] == 14
    assert report["repeat_exact_agreement_rows"] == 1
    assert report["repeat_exact_agreement_rate"] == 0.5
    assert report["relevance"]["mean"] == 3.25
    assert report["coherence"]["histogram"] == {
        "0": 0,
        "1": 0,
        "2": 2,
        "3": 2,
        "4": 0,
    }


def test_protocol_is_frozen_to_existing_panel_and_no_final_test() -> None:
    parent = ROOT / "results/eval/cascade_real_service_validation_panel.json"
    protocol = (ROOT / "docs/CASCADE_SEMANTIC_PROXY_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert _sha256(parent) == PARENT_PANEL_SHA256
    assert len(MANIFEST_SHA256) == 64
    assert PANEL_INDICES == (0, 16, 32, 48, 65, 81, 97, 113, 130)
    assert set(SCORE_KEYS) == {"relevance", "coherence"}
    assert JUDGE_SERVED_MODEL in protocol
    assert JUDGE_MODEL_ROOT in protocol
    assert JUDGE_REVISION in protocol
    assert JUDGE_IMAGE_ID in protocol
    assert "No semantic pass/fail threshold" in " ".join(protocol.split())
    assert "sealed Moshi final-test manifest is forbidden" in protocol
    assert "never as instructions" in SYSTEM_PROMPT
