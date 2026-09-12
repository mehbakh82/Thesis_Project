import numpy as np

from thesis_s2s.audio import write_wav
from thesis_s2s.bakeoff import run as bakeoff
from thesis_s2s.metrics import LatencySample


def test_bakeoff_is_component_only_and_selects_no_model(tmp_path, monkeypatch):
    clip = tmp_path / "sample.wav"
    write_wav(clip, np.zeros(1600, dtype=np.float32))
    monkeypatch.setattr(bakeoff, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bakeoff, "_load_or_synth_clips", lambda _: [clip])
    monkeypatch.setattr(
        bakeoff,
        "roundtrip_all",
        lambda _: {
            "whisper_intelligibility": {"automatic_transcript_content_preserved": True},
            "encodec_24k": {"available": True},
            "moshi_mimi": {"available": False},
            "cosyvoice2": {"available": False},
        },
    )
    monkeypatch.setattr(
        bakeoff,
        "cuda_memory_cap_gb",
        lambda _: {"device": "test", "total_gb": 24.0, "memory_capped": False},
    )
    monkeypatch.setattr(
        bakeoff,
        "measure_talker_latency",
        lambda *_: LatencySample(
            t_first_audio_ms=1.0,
            t_barge_in_ms=2.0,
            gpu_name="test",
            vram_gb=24.0,
            memory_capped=False,
        ),
    )

    report = bakeoff.run_bakeoff(tmp_path / "results", allow_hf=False)

    assert report["evidence_class"] == "component_survey"
    assert report["end_to_end_models_compared"] is False
    assert report["official"] is False
    assert report["decision"]["primary"] is None
    assert report["decision"]["status"] == "no_end_to_end_model_winner"
    assert report["decision"]["implementation_path"].startswith("moshika-7b")
    assert "moshika-7b" in report["models"]
    assert report["latency"]["official_e2e_eligible"] is False
    assert report["latency"]["t_first_audio_gate_ok"] is False
    assert report["latency"]["component_proxy_gates"]["cached_packet_p50_under_500_ms"] is True
    persisted = (tmp_path / "results" / "bakeoff_report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in persisted
    decision = (tmp_path / "results" / "DECISION.md").read_text(encoding="utf-8")
    assert "no end-to-end model winner" in decision
    assert "Locked primary" not in decision
