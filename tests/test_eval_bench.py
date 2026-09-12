import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from thesis_s2s.eval import bench


class _FakeTalker:
    backend = "fake-piper"


class _FakeSession:
    def __init__(self, _detector, _talker):
        self.log = SimpleNamespace(server_generation_ms=12.0, t_barge_in_ms=None)

    def on_user_end(self, _audio):
        return np.ones(10, dtype=np.float32)

    def on_mic_while_playing(self, _audio, *, interrupt_onset: bool):
        if interrupt_onset:
            self.log.t_barge_in_ms = 8.0


def _install_latency_fakes(tmp_path: Path, monkeypatch):
    clips = [
        SimpleNamespace(audio=np.ones(160, dtype=np.float32), kind="interrupt"),
        SimpleNamespace(audio=np.ones(160, dtype=np.float32), kind="speech"),
    ]
    trained: list[dict] = []
    loaded: list[Path] = []
    monkeypatch.setattr(bench, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bench, "make_dataset", lambda **_kwargs: clips)
    monkeypatch.setattr(bench, "DuplexSession", _FakeSession)
    monkeypatch.setattr(bench, "default_talker", lambda: _FakeTalker())
    monkeypatch.setattr(bench, "piper_available", lambda: True)
    monkeypatch.setattr(bench, "cascade_first_audio", lambda _audio: {"status": "ok"})
    monkeypatch.setattr(
        bench,
        "train_and_eval",
        lambda **kwargs: trained.append(kwargs) or {"proposed": {"accuracy": 1.0}},
    )
    monkeypatch.setattr(
        bench.BargeinDetector,
        "load",
        lambda path: loaded.append(path) or SimpleNamespace(),
    )
    return trained, loaded


def test_latency_bench_path_a_is_explicitly_unofficial(tmp_path: Path, monkeypatch):
    trained, loaded = _install_latency_fakes(tmp_path, monkeypatch)
    monkeypatch.setattr(
        bench,
        "gpu_inventory",
        lambda: {
            "device": "NVIDIA H100 NVL",
            "total_gb": 93.09,
            "official_size": False,
            "path_b_possible": False,
        },
    )
    monkeypatch.setattr(
        bench,
        "cuda_memory_cap_gb",
        lambda cap: {
            "device": "NVIDIA H100 NVL",
            "total_gb": 93.09,
            "requested_cap_gb": cap,
            "applied": True,
            "memory_capped": True,
        },
    )

    payload = bench.run_latency_bench(path="a")

    assert len(trained) == 1
    assert loaded == [tmp_path / "results" / "bargein" / "bargein_gbdt.pkl"]
    assert payload["path"] == "A_h100_memory_capped"
    assert payload["official_gpu"] is False
    assert payload["hardware_eligible"] is False
    assert payload["official_e2e_eligible"] is False
    assert payload["gpu_profile"]["requested_cap_gb"] == 24.0
    assert payload["samples"][0]["vram_gb"] == 24.0
    assert payload["samples"][0]["t_barge_in_ms"] == 8.0
    assert payload["samples"][1]["t_barge_in_ms"] is None
    assert payload["cascade"] == {"status": "ok"}
    saved = json.loads(
        (tmp_path / "results" / "eval" / "latency_bench.json").read_text(encoding="utf-8")
    )
    assert saved == payload


def test_latency_bench_path_b_reports_hardware_but_never_claims_live_e2e(
    tmp_path: Path, monkeypatch
):
    trained, loaded = _install_latency_fakes(tmp_path, monkeypatch)
    model_path = tmp_path / "results" / "bargein" / "bargein_gbdt.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    monkeypatch.setattr(
        bench,
        "gpu_inventory",
        lambda: {
            "device": "NVIDIA RTX 4090",
            "total_gb": 24.0,
            "official_size": True,
            "path_b_possible": True,
        },
    )
    monkeypatch.setattr(
        bench,
        "cuda_memory_cap_gb",
        lambda _cap: pytest.fail("Path B must not apply a software memory cap"),
    )

    payload = bench.run_latency_bench(tmp_path / "custom", path="B")

    assert trained == []
    assert loaded == [model_path]
    assert payload["path"] == "B"
    assert payload["path_b_hardware_present"] is True
    assert payload["hardware_eligible"] is True
    assert payload["official_gpu"] is False
    assert payload["official_e2e_eligible"] is False
    assert payload["samples"][0]["vram_gb"] == 24.0
    assert payload["samples"][0]["memory_capped"] is False
    assert (tmp_path / "custom" / "latency_bench_path_b.json").is_file()
    assert not (tmp_path / "custom" / "latency_bench.json").exists()


def test_interrupt_bench_distinguishes_synthetic_and_recorded_evidence(
    tmp_path: Path, monkeypatch
):
    reports = iter(
        [
            {"proposed": {"accuracy": 0.9}, "recorded_eval": None},
            {
                "proposed": {"accuracy": 0.91},
                "recorded_eval": {"n": 20},
                "official_detector_eligible": False,
                "note": "independent labels still required",
            },
        ]
    )
    calls: list[dict] = []
    monkeypatch.setattr(bench, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        bench,
        "train_and_eval",
        lambda **kwargs: calls.append(kwargs) or next(reports),
    )

    synthetic = bench.run_interrupt_bench(n_per_class=7)
    recorded = bench.run_interrupt_bench(tmp_path / "other", n_per_class=9)

    assert calls[0] == {
        "n_per_class": 7,
        "seed": 3,
        "out_dir": tmp_path / "results" / "bargein",
    }
    assert synthetic["measurement_scope"] == "synthetic_proxy"
    assert synthetic["official_detector_eligible"] is False
    assert "cannot satisfy" in synthetic["note"]
    assert recorded["measurement_scope"] == "recorded_group_heldout_automatic_labels"
    assert recorded["official_detector_eligible"] is False
    assert recorded["note"] == "independent labels still required"
    assert json.loads((tmp_path / "other" / "interrupt_bench.json").read_text()) == recorded


def _install_reply_modules(monkeypatch, asr):
    pipeline_calls: list[dict] = []

    def fake_pipeline(task: str, **kwargs):
        pipeline_calls.append({"task": task, **kwargs})
        return asr

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))
    return pipeline_calls


def test_reply_wer_closes_descriptors_removes_files_and_writes_standard_json(
    tmp_path: Path, monkeypatch
):
    results = iter([{"text": "سلام"}, "not-a-dict", {"text": "حالت خوبه"}])
    asr_calls: list[tuple[str, dict]] = []

    def fake_asr(path: str, **kwargs):
        assert Path(path).is_file()
        asr_calls.append((path, kwargs))
        return next(results)

    pipeline_calls = _install_reply_modules(monkeypatch, fake_asr)
    monkeypatch.setattr(
        bench,
        "synthesize",
        lambda _text: (np.ones(160, dtype=np.float32), "fake-piper"),
    )
    monkeypatch.setattr(
        "thesis_s2s.audio.write_wav",
        lambda path, _audio: Path(path).write_bytes(b"wav"),
    )
    descriptors: list[int] = []
    paths: list[Path] = []

    def fake_mkstemp(*, suffix: str):
        path = tmp_path / f"temporary-{len(paths)}{suffix}"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        descriptors.append(descriptor)
        paths.append(path)
        return descriptor, str(path)

    monkeypatch.setattr(bench.tempfile, "mkstemp", fake_mkstemp)

    payload = bench.run_reply_wer(tmp_path / "eval")

    assert pipeline_calls == [
        {
            "task": "automatic-speech-recognition",
            "model": "openai/whisper-small",
            "device": 0,
        }
    ]
    assert len(asr_calls) == 3
    assert all(
        kwargs == {"generate_kwargs": {"language": "persian", "task": "transcribe"}}
        for _, kwargs in asr_calls
    )
    assert payload["n"] == 3
    assert payload["rows"][1]["hyp"] == ""
    assert payload["mean_wer"] is not None
    assert all(not path.exists() for path in paths)
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    raw = (tmp_path / "eval" / "reply_wer.json").read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert json.loads(raw) == payload


def test_reply_wer_failure_removes_temp_and_uses_json_null(tmp_path: Path, monkeypatch):
    paths: list[Path] = []

    def failing_asr(path: str, **_kwargs):
        paths.append(Path(path))
        raise RuntimeError("recognizer failed")

    _install_reply_modules(monkeypatch, failing_asr)
    monkeypatch.setattr(
        bench,
        "synthesize",
        lambda _text: (np.ones(160, dtype=np.float32), "fake"),
    )
    monkeypatch.setattr(
        "thesis_s2s.audio.write_wav",
        lambda path, _audio: Path(path).write_bytes(b"wav"),
    )
    monkeypatch.setattr(bench, "project_root", lambda: tmp_path)

    payload = bench.run_reply_wer()

    assert payload["n"] == 1
    assert payload["rows"] == [{"error": "recognizer failed"}]
    assert payload["mean_wer"] is None
    assert paths and all(not path.exists() for path in paths)
    raw = (tmp_path / "results" / "eval" / "reply_wer.json").read_text()
    assert '"mean_wer": null' in raw
    assert "NaN" not in raw


def test_json_load_and_summary_create_parent_and_preserve_claim_boundaries(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "payload.json"
    source.write_text('{"value": 4}', encoding="utf-8")
    assert bench.json_load(source) == {"value": 4}
    monkeypatch.setattr(bench, "project_root", lambda: tmp_path / "new-root")
    latency = {
        "talker": "PiperTalker",
        "tts_backend": "piper",
        "t_first_audio_p50_ms": 120,
        "t_barge_in_p95_ms": 80,
        "official_gpu": False,
        "path": "A_h100_memory_capped",
        "cascade": {"status": "ok"},
    }
    interrupt = {
        "proposed": {
            "accuracy": 0.91,
            "interrupt_f1": 0.89,
            "far": 0.04,
            "frr": 0.08,
            "n": 40,
            "target_ok": False,
        },
        "energy_vad_baseline": {"accuracy": 0.7, "far": 0.2},
        "recorded_eval": None,
        "recorded_target_ok": False,
        "hop_cpu_ms": 0.4,
    }

    path = bench.write_eval_summary(latency, interrupt, None)

    text = path.read_text(encoding="utf-8")
    assert path == tmp_path / "new-root" / "results" / "eval" / "SUMMARY.md"
    assert "Synthetic proxy result: **fail**" in text
    assert "official_gpu: **False**" in text
    assert "cannot satisfy an end-to-end thesis gate" in text
    assert "5–10 participants / ≥2 aged 60+" in text


def test_summary_marks_only_the_synthetic_proxy_pass(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bench, "project_root", lambda: tmp_path)
    path = bench.write_eval_summary(
        {"path": "B"},
        {"proposed": {"target_ok": True}},
        {"mean_wer": 0.5},
    )
    text = path.read_text(encoding="utf-8")
    assert "Synthetic proxy result: **pass**" in text
    assert "mean_wer=0.5 (not a MOS" in text
