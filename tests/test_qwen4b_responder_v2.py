from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_qwen4b_responder_v2 import (
    ARMS,
    DEV_INDICES,
    FINAL_INDICES,
    TEST_SESSION_HASH_COUNTS,
    arm_order,
    engineering_gate,
    summarize,
    verify_final_unlocked,
)


def _sample(base: int, candidate: int, reference: int) -> dict:
    return {
        "arms": {
            name: {
                "scores": [
                    {"relevance": score, "coherence": score},
                    {"relevance": score, "coherence": score},
                ]
            }
            for name, score in (
                ("base", base),
                ("qwen4b_v2", candidate),
                ("reference", reference),
            )
        }
    }


def test_locked_stage_sizes_and_blind_order() -> None:
    assert len(DEV_INDICES) == 7
    assert len(FINAL_INDICES) == 40
    assert len(set(FINAL_INDICES)) == 40
    assert sum(TEST_SESSION_HASH_COUNTS.values()) == 204
    order = arm_order("development", DEV_INDICES[0])
    assert order == arm_order("development", DEV_INDICES[0])
    assert set(order) == set(ARMS)


def test_unchanged_gate_passes_clear_improvement() -> None:
    samples = [_sample(1, 3, 2) for _ in DEV_INDICES]
    summary = summarize(samples, expected_calls=42)
    assert summary["judge_calls_valid"] == 42
    assert summary["paired"]["candidate_minus_base_relevance_mean"] == 2.0
    assert all(engineering_gate(summary, valid=True).values())


def test_final_unlock_fails_closed_without_exact_stage_a(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen", encoding="utf-8")
    absent = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        verify_final_unlocked(absent, protocol)
    negative = tmp_path / "negative.json"
    negative.write_text(
        json.dumps(
            {
                "stage": "development_eligibility",
                "status": "valid_negative",
                "automatic_engineering_gate": {"gate": False},
                "validity_requirements": {"valid": True},
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        verify_final_unlocked(negative, protocol)
