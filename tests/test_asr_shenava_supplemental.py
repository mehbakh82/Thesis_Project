from scripts.evaluate_asr_candidates_v1 import CURRENT_ARM, arm_summary
from scripts.evaluate_asr_shenava_supplemental_v1 import (
    CHALLENGER_ARM,
    screen_decision,
)


def _samples(character_edits: int, word_edits: int):
    rows = []
    for index in range(40):
        rows.append(
            {
                "row_hash": f"row-{index}",
                "session_hash": f"session-{index // 5}",
                "channel": "Digiato" if index < 20 else "Zoomit",
                "reference_words": 2,
                "hypothesis_words": 2 - int(word_edits > 0),
                "word_edits": word_edits,
                "wer": word_edits / 2,
                "reference_characters": 10,
                "hypothesis_characters": 10 - character_edits,
                "character_edits": character_edits,
                "cer": character_edits / 10,
                "empty_hypothesis": False,
                "elapsed_seconds": 0.1,
                "real_time_factor": 0.05,
            }
        )
    return rows


def test_material_shenava_win_requires_new_confirmation():
    incumbent = _samples(character_edits=3, word_edits=1)
    challenger = _samples(character_edits=1, word_edits=0)
    summaries = {
        CURRENT_ARM: arm_summary(incumbent),
        CHALLENGER_ARM: arm_summary(challenger),
    }

    decision = screen_decision(incumbent, challenger, summaries)

    assert decision["status"] == "new_confirmation_required"
    assert decision["challenger_advances_to_new_confirmation"] is True
    assert decision["production_promotion_allowed"] is False


def test_nonmaterial_shenava_result_retains_incumbent():
    incumbent = _samples(character_edits=2, word_edits=0)
    challenger = _samples(character_edits=2, word_edits=0)
    summaries = {
        CURRENT_ARM: arm_summary(incumbent),
        CHALLENGER_ARM: arm_summary(challenger),
    }

    decision = screen_decision(incumbent, challenger, summaries)

    assert decision["status"] == "incumbent_retained"
    assert decision["challenger_advances_to_new_confirmation"] is False
