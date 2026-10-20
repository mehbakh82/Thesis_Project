import hashlib
import importlib.util
import io
import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_live_server import _frames, _quiet, _receive_until, _Responder, _tone

from thesis_s2s.runtime import turn_end
from thesis_s2s.runtime.live import FRAME, LiveComponents, LiveConfig, build_live_app
from thesis_s2s.runtime.vad import SAMPLE_RATE, Endpointer, EnergyScorer

ROOT = Path(__file__).resolve().parents[1]


def _push(endpointer: Endpointer, audio: np.ndarray) -> list:
    events = []
    for start in range(0, audio.size, FRAME):
        events.extend(endpointer.push(audio[start : start + FRAME]))
    return events


def test_hold_extends_the_open_pause_and_resume_clears_it() -> None:
    endpointer = Endpointer(scorer=EnergyScorer(), hangover_ms=256, tentative_ms=96)
    events = _push(endpointer, np.concatenate([_quiet(0.5), _tone(0.5), _quiet(0.15, 1)]))
    assert [e.kind for e in events] == ["speech_start", "speech_pause"]
    assert endpointer.hold(640)
    events = _push(endpointer, np.concatenate([_quiet(0.35, 2), _tone(0.5)]))
    assert [e.kind for e in events] == ["speech_resume"]  # 500 ms of silence did not end it
    events = _push(endpointer, _quiet(0.5, 3))
    assert [e.kind for e in events] == ["speech_pause", "speech_end"]  # back to 256 ms


def test_hold_ends_the_turn_after_the_held_silence() -> None:
    endpointer = Endpointer(scorer=EnergyScorer(), hangover_ms=256, tentative_ms=96)
    _push(endpointer, np.concatenate([_quiet(0.5), _tone(0.5), _quiet(0.15, 1)]))
    assert endpointer.hold(640)
    events = _push(endpointer, _quiet(1.0, 2))
    ends = [e for e in events if e.kind == "speech_end"]
    assert len(ends) == 1
    silence_ms = (ends[0].decided_at_sample - ends[0].sample) / SAMPLE_RATE * 1000
    assert 640 <= silence_ms < 640 + 64
    assert not endpointer.hold(640)  # no open pause any more


def test_lexical_gate_and_last_word() -> None:
    gate = turn_end.LexicalGate()
    assert gate.p_complete(np.zeros(1), "من می‌خواستم بپرسم که") == 0.0
    assert gate.p_complete(np.zeros(1), "کتاب را") == 0.0
    assert gate.p_complete(np.zeros(1), "ساعت چنده؟") == 1.0
    assert gate.p_complete(np.zeros(1), "") == 0.0
    assert turn_end.last_word("سلام، كتاب «ي»") == "ی"


def test_smart_turn_features_match_the_reference_extractor() -> None:
    transformers = pytest.importorskip("transformers")
    extractor = transformers.WhisperFeatureExtractor(chunk_length=8)
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(3 * SAMPLE_RATE)).astype(np.float32)
    padded = np.pad(audio, (8 * SAMPLE_RATE - audio.size, 0))
    reference = extractor(
        padded,
        sampling_rate=SAMPLE_RATE,
        return_tensors="np",
        padding="max_length",
        max_length=8 * SAMPLE_RATE,
        truncation=True,
        do_normalize=True,
    ).input_features
    ours = turn_end.smart_turn_features(audio, extractor.mel_filters)
    assert ours.shape == reference.shape == (1, 80, 800)
    assert float(np.abs(ours - reference).max()) < 1e-5


def test_llm_gate_reads_the_digit_probabilities(monkeypatch) -> None:
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data))
        top = [{"token": "0", "logprob": float(np.log(0.6))}, {"token": "1", "logprob": float(np.log(0.2))}]
        body = {"choices": [{"logprobs": {"content": [{"top_logprobs": top}]}}]}
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(turn_end.urllib.request, "urlopen", fake_urlopen)
    gate = turn_end.LLMCompletenessGate(endpoint="http://llm", model="m")
    assert gate.p_complete(np.zeros(1), "من می‌خواستم") == pytest.approx(0.25)
    assert sent["messages"][0]["content"] == turn_end.COMPLETENESS_PROMPT
    assert sent["max_tokens"] == 1 and sent["temperature"] == 0
    assert gate.p_complete(np.zeros(1), "  ") == 0.0


def test_elderly_prompt_is_the_responder_v2_prompt() -> None:
    import ast

    from thesis_s2s.runtime.cascade import ELDERLY_PROMPT_V1

    # read the frozen evaluator's constant without importing it (it needs Python 3.11)
    tree = ast.parse((ROOT / "scripts" / "evaluate_responder_candidates_v2.py").read_text(encoding="utf-8"))
    frozen = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "ELDERLY_PROMPT_V1"
    )
    assert ELDERLY_PROMPT_V1 == frozen
    digest = hashlib.sha256(ELDERLY_PROMPT_V1.encode("utf-8")).hexdigest()
    assert digest == "ecaa1293680ecd0534b69b766001ea2dfd0cd3fd78b212f0e3722d0474d909b5"


class _NeverComplete:
    name = "never_complete"
    needs_transcript = False

    def p_complete(self, audio, transcript):
        return 0.0


def _gated_app(gate):
    components = LiveComponents(
        asr=lambda audio: ("سلام چطوری", None),
        responder=_Responder(),  # type: ignore[arg-type]
        tts=lambda text: _tone(0.3, 0.1),
        identity={"asr_backend": "fake"},
        scorer_factory=EnergyScorer,
        turn_end_gate=gate,
    )
    config = LiveConfig(hangover_ms=256, speculative_ms=96, turn_end_threshold=0.5, turn_end_hold_ms=640)
    return TestClient(build_live_app(components, config))


@pytest.mark.parametrize("gated", [True, False])
def test_held_pause_merges_a_mid_sentence_pause_into_one_turn(gated: bool) -> None:
    client = _gated_app(_NeverComplete() if gated else None)
    head = np.concatenate([_quiet(0.5), _tone(0.6), _quiet(0.15, 3)])
    tail = np.concatenate([_quiet(0.3, 4), _tone(0.6), _quiet(1.2, 5)])  # a 450 ms pause in all
    with client.websocket_connect("/ws") as ws:
        for frame in _frames(head, 0):
            ws.send_bytes(frame)
        time.sleep(0.3)  # the gate answers while the pause is still open
        for frame in _frames(tail, head.size // FRAME):
            ws.send_bytes(frame)
        seen = _receive_until(ws, lambda p: p.get("event") == "endpoint")
        endpoint = [p for kind, p in seen if kind == "json" and p["event"] == "endpoint"][0]
    if gated:  # one turn, ending after the second fragment
        assert abs(endpoint["end_sample"] - int(2.15 * SAMPLE_RATE)) <= 1024
    else:  # cut after the first fragment
        assert abs(endpoint["end_sample"] - int(1.1 * SAMPLE_RATE)) <= 1024


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_comparison_uses_an_exact_mcnemar_test() -> None:
    splits = _script("compare_loopback_splits")
    assert splits.mcnemar_exact(0, 0) == 1.0
    assert splits.mcnemar_exact(6, 0) == pytest.approx(2 / 64)
    before = {"a": True, "b": True, "c": False, "d": True}
    after = {"a": False, "b": True, "c": True, "d": False}
    result = splits.compare(before, after)
    assert (result["split_before"], result["split_after"]) == (3, 2)
    assert (result["split_only_before"], result["split_only_after"]) == (2, 1)


def test_development_selection_rule(tmp_path) -> None:
    dev = _script("end_of_turn_dev")

    def rows(premature: int, held: int, n: int = 100) -> list[dict]:
        return [
            {"trial": str(i), "condition": "plain", "premature": i < premature,
             "final_held": premature <= i < premature + held, "added_delay_ms": 0.0,
             "late_holds": 0, "unended": False}
            for i in range(n)
        ]

    (tmp_path / "trials.jsonl").write_text("{}\n")
    (tmp_path / "replay_none.json").write_text(json.dumps({"none|0.0|256": rows(20, 0)}))
    (tmp_path / "replay_lexical.json").write_text(json.dumps({
        "lexical|1.0|640": rows(12, 4),   # eligible, -40%
        "lexical|1.0|1024": rows(12, 3),  # tie on premature, fewer false holds -> chosen
    }))
    (tmp_path / "replay_smart_turn_v3_2.json").write_text(json.dumps({"smart_turn_v3_2|0.5|640": rows(5, 9)}))
    out = tmp_path / "receipt.json"
    args = type("A", (), {"work": tmp_path, "out": out, "stage2_training": None, "invalid_replay": None})()
    assert dev.select(args) == 0
    receipt = json.loads(out.read_text())
    assert receipt["adopted"] and receipt["chosen"]["hold_ms"] == 1024
    assert receipt["relative_reduction"] == pytest.approx(0.4)
    with pytest.raises(SystemExit):
        dev.select(args)  # write-once


def test_responder_switch_qualification_checks() -> None:
    switch = _script("score_responder_switch")
    loopback = {
        "conditions": {
            "plain": {"turns_without_first_audio": 0, "t_first_audio_ms": {"mean": 410.0},
                      "t_first_audio_p90_failures_as_late_ms": 470.0},
            "elderly": {"t_first_audio_p90_failures_as_late_ms": 480.0},
        },
        "acceptance": {"plain_mean_le_500ms": True, "plain_p90_le_500ms_failures_counted": True,
                       "elderly_p90_le_500ms_failures_counted": True},
        "gpu_memory": {"peak_stack_mib": 11300},
    }
    coherence = {"criterion": {"per_judge": {"gemini": True, "qwen38": True}, "met": True}}
    result = switch.qualification(loopback, coherence, ["سلام عزیزم", "بله"])
    assert result["part2_passed"]
    result = switch.qualification(loopback, coherence, ["سلام OK"])
    assert not result["checks"]["q4_script"]["passed"] and not result["part2_passed"]
    loopback["gpu_memory"]["peak_stack_mib"] = 12400
    assert not switch.qualification(loopback, coherence, [])["checks"]["q1_memory"]["passed"]
