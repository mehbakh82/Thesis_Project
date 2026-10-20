import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_listening_study.py"
SPEC = importlib.util.spec_from_file_location("ingest_listening_study", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ITEMS = [{"id": f"p{i}"} for i in range(8)]
NEW, OLD = MODULE.CHALLENGER, MODULE.INCUMBENT


def _participant(age: str, new: int, old: int) -> dict:
    picks = [NEW] * new + [OLD] * old + ["tie"] * (8 - new - old)
    return {"age": age, "preferences": {it["id"]: p for it, p in zip(ITEMS, picks, strict=True)}}


def test_switch_rule_passes_on_a_consistent_preference():
    accepted = [
        _participant("under_40", 6, 1),
        _participant("40_59", 5, 2),
        _participant("under_40", 7, 0),
        _participant("60plus", 5, 2),
        _participant("60plus", 4, 3),
    ]
    rule = MODULE.responder_switch_rule(accepted, ITEMS)
    assert rule["sample_requirement_met"] and rule["preference_interval_above_zero"]
    assert rule["human_preference_met"]
    assert abs(rule["mean_preference_difference"] - (5 + 3 + 7 + 3 + 1) / 40) < 1e-9


def test_switch_rule_fails_if_an_older_participant_prefers_the_incumbent():
    accepted = [
        _participant("under_40", 8, 0),
        _participant("40_59", 8, 0),
        _participant("under_40", 8, 0),
        _participant("60plus", 6, 1),
        _participant("60plus", 2, 4),
    ]
    rule = MODULE.responder_switch_rule(accepted, ITEMS)
    assert rule["preference_interval_above_zero"]
    assert not rule["no_60_plus_participant_prefers_incumbent"]
    assert not rule["human_preference_met"]


def test_switch_rule_needs_the_definition_sample():
    accepted = [_participant("under_40", 8, 0)] * 4 + [_participant("60plus", 8, 0)]
    rule = MODULE.responder_switch_rule(accepted, ITEMS)
    assert not rule["sample_requirement_met"] and not rule["human_preference_met"]


def test_switch_rule_fails_on_a_split_preference():
    accepted = [
        _participant("under_40", 6, 1),
        _participant("40_59", 1, 6),
        _participant("under_40", 4, 4),
        _participant("60plus", 4, 3),
        _participant("60plus", 3, 3),
    ]
    rule = MODULE.responder_switch_rule(accepted, ITEMS)
    assert not rule["preference_interval_above_zero"] and not rule["human_preference_met"]
