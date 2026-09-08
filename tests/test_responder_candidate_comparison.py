import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_responder_candidates_v1.py"
SPEC = importlib.util.spec_from_file_location("responder_candidates_v1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _rows():
    rows = []
    for channel in MODULE.CHANNELS:
        for session_index in range(5):
            for row_index in range(6):
                rows.append(
                    {
                        "utt_id": f"{channel}-{session_index}-{row_index}",
                        "session_id": f"{channel}/session-{session_index}",
                        "split": "train",
                        "text": "این یک پرسش فارسی برای آزمون است",
                        "response_text": "این یک پاسخ فارسی مناسب برای آزمون است",
                    }
                )
    return rows


def test_panel_selection_is_equal_channel_and_session_disjoint():
    panels = MODULE.select_panels(_rows())

    assert len(panels["development"]) == MODULE.EXPECTED_ROWS
    assert len(panels["final"]) == MODULE.EXPECTED_ROWS
    for stage in ("development", "final"):
        receipt = MODULE.panel_receipt(panels[stage])
        assert receipt["sessions"] == 8
        assert set(receipt["channels"].values()) == {10}
    development_sessions = {row["session_id"] for row in panels["development"]}
    final_sessions = {row["session_id"] for row in panels["final"]}
    assert development_sessions.isdisjoint(final_sessions)


def test_selection_retains_incumbent_without_material_gain():
    def sample(row_hash, relevance, coherence):
        return {
            "row_hash": row_hash,
            "scores": [
                {"relevance": relevance, "coherence": coherence},
                {"relevance": relevance, "coherence": coherence},
            ],
        }

    incumbent = [sample(str(index), 3, 3) for index in range(4)]
    challenger = [sample(str(index), 3, 3) for index in range(4)]
    summaries = {
        MODULE.CURRENT_ARM: {
            "mechanically_valid": True,
            "relevance": {"mean": 3.0},
            "coherence": {"mean": 3.0},
            "generation_ms": {"p50": 10.0},
        },
        "qwen3.5-4b": {
            "mechanically_valid": True,
            "relevance": {"mean": 3.0},
            "coherence": {"mean": 3.0},
            "generation_ms": {"p50": 5.0},
        },
    }
    result = MODULE.select_development_winner(
        {MODULE.CURRENT_ARM: incumbent, "qwen3.5-4b": challenger}, summaries
    )

    assert result["selected_for_final"] == MODULE.CURRENT_ARM
    assert result["status"] == "incumbent_retained"


def test_selection_promotes_only_material_challenger():
    def sample(row_hash, relevance, coherence):
        return {
            "row_hash": row_hash,
            "scores": [
                {"relevance": relevance, "coherence": coherence},
                {"relevance": relevance, "coherence": coherence},
            ],
        }

    incumbent = [sample(str(index), 2, 3) for index in range(4)]
    challenger = [sample(str(index), 3, 3) for index in range(4)]
    summaries = {
        MODULE.CURRENT_ARM: {
            "mechanically_valid": True,
            "relevance": {"mean": 2.0},
            "coherence": {"mean": 3.0},
            "generation_ms": {"p50": 10.0},
        },
        "qwen3.5-4b": {
            "mechanically_valid": True,
            "relevance": {"mean": 3.0},
            "coherence": {"mean": 3.0},
            "generation_ms": {"p50": 12.0},
        },
    }
    result = MODULE.select_development_winner(
        {MODULE.CURRENT_ARM: incumbent, "qwen3.5-4b": challenger}, summaries
    )

    assert result["selected_for_final"] == "qwen3.5-4b"
    assert result["status"] == "challenger_promoted"
