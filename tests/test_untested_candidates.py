"""Scoring logic of the ASR v3, responder size and end-of-turn v2 experiments."""

import importlib.util
import random
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIZE = _load("evaluate_responder_size")
EOT = _load("end_of_turn_dev")
SMART = _load("smart_turn_persian")


def _row(trial: str, held: bool, premature: bool = False) -> dict:
    condition = "plain" if trial.startswith("plain/") else "simulated_elderly"
    return {"trial": trial, "condition": condition, "premature": premature, "final_held": held,
            "added_delay_ms": 384.0 if held else 0.0, "late_holds": 0, "unended": False}


def test_v2_false_holds_count_only_labelled_complete_ends():
    rows = [
        _row("plain/a", held=True),        # complete end held: a false hold
        _row("plain/b", held=False),       # complete end released
        _row("plain/c", held=True),        # unfinished end held: a correct hold
        _row("elderly/c", held=True),
        _row("plain/d", held=True, premature=True),  # uncertain label: in neither denominator
    ]
    labels = {"a": "complete", "b": "complete", "c": "incomplete", "d": "uncertain"}
    entry = EOT.summarize_v2({"lexical|1.0|640": rows}, labels)["lexical|1.0|640"]
    assert entry["complete_end_trials"] == 2 and entry["false_hold_rate"] == 0.5
    assert entry["incomplete_end_trials"] == 2 and entry["correct_hold_rate"] == 1.0
    assert entry["false_hold_rate_v1_all_trials"] == 4 / 5
    assert entry["premature_rate"] == 1 / 5


def test_window_turns_merge_one_speakers_consecutive_segments():
    window = {"exclusive_speaker_turns": str([
        {"start": 0.0, "end": 2.0, "speaker": "A"},
        {"start": 2.4, "end": 5.0, "speaker": "A"},
        {"start": 5.6, "end": 7.0, "speaker": "B"},
        {"start": 7.5, "end": 9.0, "speaker": "A"},
    ])}
    turns = SMART._turns(window)
    assert [(t["speaker"], t["start"], t["end"]) for t in turns] == [("A", 0.0, 5.0), ("B", 5.6, 7.0), ("A", 7.5, 9.0)]


def _table(gains: list[int]) -> dict:
    table = {}
    for i, gain in enumerate(gains):
        table[("ref", f"r{i}")] = {"relevance": 2, "coherence": 3, "elderly_appropriateness": 3}
        table[("big", f"r{i}")] = {"relevance": 2 + gain, "coherence": 3, "elderly_appropriateness": 3}
    return table


def test_size_rule_needs_gain_and_row_wins():
    rows = [f"r{i}" for i in range(20)]
    strong = SIZE.paired(_table([1] * 12 + [0] * 8), "big", "ref", rows, random.Random(1))
    assert strong["relevance_gain"] == 0.6 and strong["row_win_rate"] == 0.6
    assert SIZE.meets(strong)
    few_wins = SIZE.paired(_table([3] * 5 + [0] * 15), "big", "ref", rows, random.Random(1))
    assert few_wins["relevance_gain"] == 0.75 and not SIZE.meets(few_wins)  # a large mean from 25% of rows
    worse_coherence = dict(strong, coherence_gain=-0.05)
    assert not SIZE.meets(worse_coherence)


def test_size_arms_are_the_registered_set():
    assert SIZE.REFERENCE == "qwen35_4b_elderly_v1" and SIZE.INCUMBENT == "qwen3_4b_prompt_v2"
    assert set(SIZE.LARGER) <= set(SIZE.ARMS) and len(SIZE.ARMS) == 8
