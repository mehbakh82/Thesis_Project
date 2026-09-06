from __future__ import annotations

from scripts.evaluate_qwen4b_responder import (
    EXPECTED_JUDGE_CALLS,
    EXPECTED_ROWS,
    MODEL_BYTES,
    MODEL_FILE_COUNT,
    MODEL_TREE_SHA256,
    PANEL_INDICES,
    PRIOR_INDICES,
    RESIDUAL_INDICES,
    arm_order,
    engineering_gate,
    summarize,
)


def _sample(base: int, qwen4b: int, reference: int) -> dict:
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
                ("qwen4b", qwen4b),
                ("reference", reference),
            )
        }
    }


def test_panel_is_new_fixed_and_leaves_residual_rows() -> None:
    assert len(PANEL_INDICES) == EXPECTED_ROWS == 40
    assert len(set(PANEL_INDICES)) == EXPECTED_ROWS
    assert not set(PANEL_INDICES) & set(PRIOR_INDICES)
    assert not set(RESIDUAL_INDICES) & (set(PANEL_INDICES) | set(PRIOR_INDICES))
    assert EXPECTED_JUDGE_CALLS == 240


def test_model_lock_and_blind_order_are_fixed() -> None:
    assert len(MODEL_TREE_SHA256) == 64
    assert MODEL_FILE_COUNT == 28
    assert MODEL_BYTES == 8_060_919_167
    order = arm_order(PANEL_INDICES[0])
    assert order == arm_order(PANEL_INDICES[0])
    assert set(order) == {"base", "qwen4b", "reference"}


def test_summary_and_gate_pass_for_working_candidate() -> None:
    summary = summarize([_sample(1, 3, 2) for _ in range(EXPECTED_ROWS)])
    assert summary["judge_calls_valid"] == EXPECTED_JUDGE_CALLS
    assert summary["qwen4b"]["relevance"]["mean"] == 3.0
    assert summary["paired"]["qwen4b_minus_base_relevance_mean"] == 2.0
    assert summary["paired"]["qwen4b_relevance_win_rate"] == 1.0
    assert all(engineering_gate(summary, valid=True).values())


def test_gate_fails_when_candidate_does_not_improve() -> None:
    summary = summarize([_sample(2, 2, 2) for _ in range(EXPECTED_ROWS)])
    gate = engineering_gate(summary, valid=True)
    assert gate["relevance_gain_at_least_0_50"] is False
    assert gate["relevance_win_rate_at_least_0_60"] is False
    assert not all(gate.values())
