import json
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.audio import read_wav
from thesis_s2s.bargein.synthetic import _harmonic
from thesis_s2s.data.filter_corpus import filter_hours
from thesis_s2s.runtime.duplex import DummyTalker, build_app
from thesis_s2s.runtime.session_log import PROMPTS, SessionMeta, SessionStore
from thesis_s2s.runtime.tts import FormantTalker, piper_available


def test_session_kit_writes_wav_and_jsonl(tmp_path: Path):
    store = SessionStore(root=tmp_path)
    meta = SessionMeta(session_id="Sdemo", speaker_id="P60", age_bin="60plus", consent=True)
    store.start(meta)
    audio = _harmonic(0.4, 180)
    rec = store.add_turn(
        meta,
        prompt_id="interrupt_story",
        interrupt_label="interrupt",
        user_audio=audio,
        t_first_audio_ms=120.0,
        t_barge_in_ms=40.0,
        stopped=True,
    )
    wav, sr = read_wav(rec.audio_filepath)
    assert sr == 16000
    assert len(wav) > 100
    report = store.export_manifest(tmp_path / "recorded.jsonl")
    assert report["n"] == 1
    assert report["elderly_turns"] == 1
    assert "interrupt" in rec.interrupt_label
    assert any(p["id"] == "backchannel" for p in PROMPTS)


def test_session_identity_and_turn_enums_fail_closed(tmp_path: Path):
    store = SessionStore(root=tmp_path)
    with pytest.raises(ValueError, match="speaker_id"):
        store.start(SessionMeta("S1", "../speaker", "under_60", True))
    with pytest.raises(ValueError, match="age_bin"):
        store.start(SessionMeta("S1", "P1", "unknown", True))
    with pytest.raises(ValueError, match="retention"):
        store.start(SessionMeta("S1", "P1", "under_60", True, retention="raw"))

    meta = SessionMeta("S1", "P1", "under_60", True, retention="metrics")
    store.start(meta)
    with pytest.raises(ValueError, match="different speaker"):
        store.start(SessionMeta("S1", "P2", "under_60", True, retention="metrics"))
    with pytest.raises(ValueError, match="unknown prompt_id"):
        store.add_turn(
            meta,
            prompt_id="unregistered",
            interrupt_label="none",
            user_audio=np.zeros(160, dtype=np.float32),
            t_first_audio_ms=None,
            t_barge_in_ms=None,
            stopped=False,
        )
    with pytest.raises(ValueError, match="unknown interrupt_label"):
        store.add_turn(
            meta,
            prompt_id="warmup_time",
            interrupt_label="maybe",
            user_audio=np.zeros(160, dtype=np.float32),
            t_first_audio_ms=None,
            t_barge_in_ms=None,
            stopped=False,
        )


def test_study_summary_requires_and_recognizes_all_strict_gates(
    tmp_path: Path, monkeypatch
) -> None:
    from thesis_s2s.runtime import session_log

    monkeypatch.setattr(session_log, "project_root", lambda: tmp_path)
    store = SessionStore(root=tmp_path / "recordings")
    audio = np.zeros(1600, dtype=np.float32)
    for index in range(5):
        speaker_id = f"P{index}"
        meta = SessionMeta(
            session_id=f"S{index}",
            speaker_id=speaker_id,
            age_bin="60plus" if index < 2 else "under_60",
            consent=True,
            gpu_name="NVIDIA GeForce RTX 4090",
            vram_gb=24.0,
            retention="metrics",
        )
        is_interrupt = index % 2 == 0
        store.add_turn(
            meta,
            prompt_id="interrupt_story" if is_interrupt else "warmup_time",
            interrupt_label="interrupt" if is_interrupt else "none",
            user_audio=audio,
            t_first_audio_ms=200.0 + index,
            t_barge_in_ms=80.0 + index,
            stopped=is_interrupt,
        )
        session_dir = store.session_dir(meta.session_id)
        (session_dir / "mos.jsonl").write_text(
            json.dumps(
                {
                    "session_id": meta.session_id,
                    "speaker_id": speaker_id,
                    "age_bin": meta.age_bin,
                    "naturalness": 4,
                    "latency": 4,
                    "interrupt_success": 5,
                    "satisfaction": 4,
                    "consent": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    detector_dir = tmp_path / "results" / "bargein"
    detector_dir.mkdir(parents=True)
    (detector_dir / "recorded_heldout_report.json").write_text(
        json.dumps(
            {
                "evidence_class": "recorded_audio_heldout",
                "n": 25,
                "group_overlap": False,
                "proposed": {"target_ok": True, "accuracy": 0.84},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "study_summary.json"
    report = store.study_summary(out)

    assert report["status"] == "complete"
    assert report["official_ready"] is True
    assert report["participants"] == 5
    assert report["elderly_participants"] == 2
    assert report["complete_ratings"] == 5
    assert report["official_live_interrupt"]["accuracy"] == 1.0
    assert report["retention_counts"] == {"audio": 0, "features": 0, "metrics": 5}
    assert all(report["requirements"].values())
    assert json.loads(out.read_text(encoding="utf-8"))["official_ready"] is True

    invalid_path = store.session_dir("S0") / "mos.jsonl"
    with invalid_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "session_id": "S0",
                    "speaker_id": "P0",
                    "age_bin": "60plus",
                    "naturalness": 6,
                    "latency": 4,
                    "interrupt_success": 5,
                    "satisfaction": 4,
                    "consent": True,
                }
            )
            + "\n"
        )
    rejected = store.study_summary()
    assert rejected["invalid_rating_rows"] == 1
    assert rejected["requirements"]["rating_rows_valid"] is False
    assert rejected["official_ready"] is False


def test_filter_require_teacher(tmp_path: Path):
    import json

    jsonl = tmp_path / "in.jsonl"
    rows = [
        {
            "duration": 3.0,
            "text": "این یک جمله فارسی آزمایشی است",
            "transcript_nemo": "",
            "snr": 15.0,
        },
        {
            "duration": 3.0,
            "text": "این یک جمله فارسی آزمایشی است",
            "transcript_nemo": "این یک جمله فارسی آزمایشی است",
            "snr": 15.0,
        },
    ]
    jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    cap = filter_hours(
        jsonl,
        tmp_path / "cap.jsonl",
        min_hours=0,
        max_hours=1,
        compute_snr=False,
        require_teacher=False,
    )
    tea = filter_hours(
        jsonl,
        tmp_path / "tea.jsonl",
        min_hours=0,
        max_hours=1,
        compute_snr=False,
        require_teacher=True,
    )
    assert cap["n"] == 2
    assert tea["n"] == 1
    assert tea["require_teacher"] is True


def test_study_rating_endpoint(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from thesis_s2s.bargein.detector import EnergyVadBaseline
    from thesis_s2s.runtime import duplex as duplex_mod
    from thesis_s2s.runtime import session_log

    monkeypatch.setattr(session_log, "project_root", lambda: tmp_path)
    monkeypatch.setattr(duplex_mod, "default_talker", lambda: DummyTalker())
    monkeypatch.setattr("thesis_s2s.config.project_root", lambda: tmp_path)
    app = build_app(
        EnergyVadBaseline(), record=True, study=True, session_id="Srate", age_bin="60plus"
    )
    client = TestClient(app)
    r = client.post(
        "/study/rating",
        json={
            "naturalness": 4,
            "interrupt_success": 5,
            "elderly_notes": "pause ok",
            "consent": True,
        },
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    h = client.get("/health")
    denied = client.post("/study/rating", json={"naturalness": 4, "consent": False})
    assert denied.status_code == 403
    invalid = client.post("/study/rating", json={"naturalness": 6, "consent": True})
    assert invalid.status_code == 422
    assert h.json()["record"] is True
    p = client.get("/prompts")
    assert len(p.json()["prompts"]) >= 4


def test_websocket_records_client_observed_timing(tmp_path: Path, monkeypatch):
    import numpy as np
    from fastapi.testclient import TestClient

    from thesis_s2s.bargein.detector import EnergyVadBaseline
    from thesis_s2s.runtime import duplex as duplex_mod
    from thesis_s2s.runtime import session_log

    monkeypatch.setattr(session_log, "project_root", lambda: tmp_path)
    monkeypatch.setattr(duplex_mod, "default_talker", lambda: DummyTalker())
    app = build_app(EnergyVadBaseline(), record=True, session_id="Sws", speaker_id="Pws")
    client = TestClient(app)
    audio = _harmonic(0.3, 190)
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    meta = {
        "session_id": "Sws",
        "speaker_id": "Pws",
        "age_bin": "under_60",
        "prompt_id": "warmup_time",
        "interrupt_label": "none",
        "consent": True,
    }
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"event": "hello", **meta})
        ws.send_json({"event": "begin_utterance", **meta})
        ws.send_bytes(pcm.tobytes())
        ws.send_json({"event": "end_of_speech", **meta})
        reply = ws.receive_bytes()
        ready = ws.receive_json()
        assert len(reply) > 4000
        assert ready["event"] == "reply_ready"
        ws.send_json({"event": "playback_started", "t_first_audio_ms": 123.0, **meta})
        ws.send_json({"event": "playback_ended", **meta})
    turns = (tmp_path / "data" / "recordings" / "Sws" / "turns.jsonl").read_text(encoding="utf-8")
    assert json.loads(turns)["t_first_audio_ms"] == 123.0


def test_piper_flag_does_not_crash():
    talker = FormantTalker()
    audio = talker.first_chunk(_harmonic(0.3, 160), "سلام")
    assert len(audio) > 100
    assert talker.backend in {"piper", "formant"}
    assert piper_available() in {True, False}
