import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_asr_candidates_v1.py"
SPEC = importlib.util.spec_from_file_location("evaluate_asr_candidates_v1", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _row(channel: str, session: int, row: int) -> dict:
    return {
        "split": "train",
        "session_id": f"{channel}/session-{session}",
        "utt_id": f"{channel}-{session}-{row}",
        "text": "این یک متن آزمایشی فارسی است",
        "user_duration": 4.0,
        "audio_filepath": __file__,
    }


def test_panels_are_balanced_and_session_disjoint() -> None:
    rows = [
        _row(channel, session, row)
        for channel in module.CHANNELS
        for session in range(9)
        for row in range(6)
    ]
    panels = module.select_panels(rows)
    assert all(len(panel) == 40 for panel in panels.values())
    for panel in panels.values():
        assert module.panel_receipt(panel)["channels"] == {"Digiato": 20, "Zoomit": 20}
    development = {row["session_id"] for row in panels["development"]}
    final = {row["session_id"] for row in panels["final"]}
    assert development.isdisjoint(final)


def _summary(cer: float, wer: float, rtf: float = 0.5) -> dict:
    return {
        "mechanically_valid": True,
        "character_error_rate": {"micro": cer},
        "word_error_rate": {"micro": wer},
        "real_time_factor": {"p50": rtf},
    }


def test_selection_retains_incumbent_without_material_gain(monkeypatch) -> None:
    summaries = {
        module.CURRENT_ARM: _summary(0.20, 0.30),
        "qwen3-asr-1.7b": _summary(0.19, 0.29),
        "qwen3-asr-0.6b": _summary(0.25, 0.35),
    }
    monkeypatch.setattr(
        module,
        "paired_comparison",
        lambda old, new, arm: {
            "cer_absolute_reduction": 0.01 if arm.endswith("1.7b") else -0.05,
            "wer_absolute_reduction": 0.01 if arm.endswith("1.7b") else -0.05,
            "challenger_minus_incumbent_cer_bootstrap_95": [-0.03, 0.01],
        },
    )
    result = module.select_winner({arm: [] for arm in module.ALL_ARMS}, summaries)
    assert result["status"] == "incumbent_retained"
    assert result["selected_for_final"] == module.CURRENT_ARM


def test_selection_promotes_only_clear_challenger(monkeypatch) -> None:
    summaries = {
        module.CURRENT_ARM: _summary(0.25, 0.35),
        "qwen3-asr-1.7b": _summary(0.15, 0.25),
        "qwen3-asr-0.6b": _summary(0.30, 0.40),
    }
    monkeypatch.setattr(
        module,
        "paired_comparison",
        lambda old, new, arm: {
            "cer_absolute_reduction": 0.10 if arm.endswith("1.7b") else -0.05,
            "wer_absolute_reduction": 0.10 if arm.endswith("1.7b") else -0.05,
            "challenger_minus_incumbent_cer_bootstrap_95": [-0.14, -0.04]
            if arm.endswith("1.7b")
            else [0.01, 0.09],
        },
    )
    result = module.select_winner({arm: [] for arm in module.ALL_ARMS}, summaries)
    assert result["status"] == "challenger_promoted"
    assert result["selected_for_final"] == "qwen3-asr-1.7b"


def test_empty_hypothesis_remains_in_error_denominator(monkeypatch) -> None:
    monkeypatch.setattr(module, "EXPECTED_ROWS", 2)
    common = {
        "session_hash": "session",
        "reference_words": 3,
        "reference_characters": 6,
        "hypothesis_words": 3,
        "hypothesis_characters": 6,
        "word_edits": 0,
        "character_edits": 0,
        "wer": 0.0,
        "cer": 0.0,
        "real_time_factor": 0.5,
        "empty_hypothesis": False,
    }
    empty = {
        **common,
        "reference_words": 2,
        "reference_characters": 4,
        "hypothesis_words": 0,
        "hypothesis_characters": 0,
        "word_edits": 2,
        "character_edits": 4,
        "wer": 1.0,
        "cer": 1.0,
        "empty_hypothesis": True,
    }
    summary = module.arm_summary([common, empty])
    assert summary["character_error_rate"]["reference_characters"] == 10
    assert summary["character_error_rate"]["edits"] == 4
    assert summary["empty_hypotheses"] == 1
    assert summary["mechanically_valid"] is True
