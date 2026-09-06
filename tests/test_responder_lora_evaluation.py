from __future__ import annotations

from scripts.evaluate_responder_lora import (
    EXPECTED_FINAL_ROWS,
    EXPECTED_JUDGE_CALLS,
    FINAL_PANEL_INDICES,
    PREVIOUSLY_OBSERVED_INDICES,
    blind_arm_order,
    engineering_gate,
    summarize,
)


def _sample(base: int, lora: int, reference: int) -> dict:
    return {
        "arms": {
            "base": {
                "scores": [
                    {"relevance": base, "coherence": base},
                    {"relevance": base, "coherence": base},
                ]
            },
            "lora": {
                "scores": [
                    {"relevance": lora, "coherence": lora},
                    {"relevance": lora, "coherence": lora},
                ]
            },
            "reference": {
                "scores": [
                    {"relevance": reference, "coherence": reference},
                    {"relevance": reference, "coherence": reference},
                ]
            },
        }
    }


def test_locked_panel_excludes_previously_observed_rows() -> None:
    assert len(FINAL_PANEL_INDICES) == EXPECTED_FINAL_ROWS == 40
    assert len(set(FINAL_PANEL_INDICES)) == EXPECTED_FINAL_ROWS
    assert not set(FINAL_PANEL_INDICES) & set(PREVIOUSLY_OBSERVED_INDICES)
    assert EXPECTED_JUDGE_CALLS == 240


def test_blind_arm_order_is_deterministic_and_complete() -> None:
    first = blind_arm_order(FINAL_PANEL_INDICES[0])
    assert first == blind_arm_order(FINAL_PANEL_INDICES[0])
    assert set(first) == {"base", "lora", "reference"}


def test_summary_and_predeclared_gate_pass_for_clear_improvement() -> None:
    samples = [_sample(1, 3, 4) for _ in range(EXPECTED_FINAL_ROWS)]
    summary = summarize(samples)
    assert summary["judge_calls_valid"] == EXPECTED_JUDGE_CALLS
    assert summary["base"]["relevance"]["mean"] == 1.0
    assert summary["lora"]["relevance"]["mean"] == 3.0
    assert summary["paired"]["lora_minus_base_relevance_mean"] == 2.0
    assert summary["paired"]["lora_relevance_win_rate"] == 1.0
    assert all(engineering_gate(summary, mechanically_valid=True).values())


def test_gate_fails_when_scores_are_valid_but_not_improved() -> None:
    samples = [_sample(2, 2, 3) for _ in range(EXPECTED_FINAL_ROWS)]
    summary = summarize(samples)
    gate = engineering_gate(summary, mechanically_valid=True)
    assert gate["mechanically_valid"] is True
    assert gate["relevance_gain_at_least_0_50"] is False
    assert gate["relevance_win_rate_at_least_0_60"] is False
    assert not all(gate.values())
