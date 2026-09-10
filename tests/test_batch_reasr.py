import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from thesis_s2s.data import batch_reasr


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_load_speech_service_is_explicit_and_fail_closed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("THESIS_NEMO_ASR_API_ROOT", raising=False)
    assert batch_reasr._load_speech_service() is None

    monkeypatch.setenv("THESIS_NEMO_ASR_API_ROOT", str(tmp_path / "missing"))
    assert batch_reasr._load_speech_service() is None

    asr_engine = ModuleType("asr_engine")
    service_module = ModuleType("service")
    pipeline_module = ModuleType("speech_pipeline")
    transcribe_chunks = lambda *_args, **_kwargs: {}  # noqa: E731
    audio_slice = lambda *_args, **_kwargs: np.zeros(1)  # noqa: E731

    class SpeechService:
        pass

    asr_engine.transcribe_chunks = transcribe_chunks
    service_module.SpeechService = SpeechService
    pipeline_module.audio_slice = audio_slice
    monkeypatch.setitem(sys.modules, "asr_engine", asr_engine)
    monkeypatch.setitem(sys.modules, "service", service_module)
    monkeypatch.setitem(sys.modules, "speech_pipeline", pipeline_module)
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    monkeypatch.setenv("THESIS_NEMO_ASR_API_ROOT", str(tmp_path))

    assert batch_reasr._load_speech_service() == (
        SpeechService,
        transcribe_chunks,
        audio_slice,
    )


def test_reasr_one_handles_rejection_and_normalizes_success(tmp_path: Path, monkeypatch) -> None:
    audio = np.arange(160, dtype=np.float32)
    monkeypatch.setattr(batch_reasr, "read_wav", lambda _path: (audio, 16_000))

    rejected = SimpleNamespace(language_status="not_fa", chunks=[])
    service = SimpleNamespace(prepare=lambda *_args, **_kwargs: rejected)
    assert batch_reasr.reasr_one(service, None, None, tmp_path / "x.wav") == {
        "transcript_nemo": None,
        "language_status": "not_fa",
        "alignment": [],
        "teacher": "nemo-soroush",
    }

    prepared = SimpleNamespace(
        language_status="ok",
        chunks=[SimpleNamespace(start=0.1, end=0.2)],
        asr_audio=audio,
        enhancement="disabled",
    )
    service.prepare = lambda *_args, **_kwargs: prepared
    calls: dict[str, object] = {}

    def audio_slice(source, start, end, sample_rate):
        calls["slice"] = (source, start, end, sample_rate)
        return source[1:3]

    def transcribe_chunks(chunks, starts, **kwargs):
        calls["transcribe"] = (chunks, starts, kwargs)
        return {"persian": "  سلام  كجايي  ", "persian_alignment": [{"start": 0.1}]}

    result = batch_reasr.reasr_one(
        service, transcribe_chunks, audio_slice, tmp_path / "x.wav"
    )
    assert result == {
        "transcript_nemo": "سلام کجایی",
        "language_status": "ok",
        "alignment": [{"start": 0.1}],
        "enhancement": "disabled",
        "teacher": "nemo-soroush",
    }
    assert calls["slice"][1:] == (0.1, 0.2, 16_000)
    assert calls["transcribe"][1:] == (
        [0.1],
        {"sample_rate": 16_000, "enable_alignment": True, "use_ngram": False},
    )


def test_teacher_adapters_preserve_verbatim_persian_and_errors(tmp_path: Path, monkeypatch) -> None:
    wav = tmp_path / "sample.wav"
    wav.touch()

    whisper = object.__new__(batch_reasr.WhisperTeacher)
    whisper.model_name = "fake-whisper"
    whisper.pipe = lambda *_args, **_kwargs: {"text": "  سلام  كجايي "}
    assert whisper.transcribe(wav)["language_status"] == "ok"
    whisper.pipe = lambda *_args, **_kwargs: {"text": "hello"}
    assert whisper.transcribe(wav)["language_status"] == "language_uncertain"

    faster = object.__new__(batch_reasr.FasterWhisperTeacher)
    faster.model_name = "fake-faster"
    faster.model = SimpleNamespace(
        transcribe=lambda *_args, **_kwargs: (
            [SimpleNamespace(text=" سلام"), SimpleNamespace(text=" خوبی")],
            object(),
        )
    )
    faster_result = faster.transcribe(wav)
    assert faster_result["transcript_nemo"] == "سلام خوبی"
    assert faster_result["teacher"] == "fake-faster"

    monkeypatch.setattr(batch_reasr, "read_wav", lambda _path: (np.zeros(16), 16_000))
    monkeypatch.setattr("thesis_s2s.runtime.cascade._wav_bytes", lambda _audio: b"wav")
    monkeypatch.setattr(
        "thesis_s2s.runtime.cascade.asr_http",
        lambda payload: {
            "persian": "سلام",
            "persian_alignment": [{"word": "سلام"}],
            "payload": payload,
        },
    )
    http = batch_reasr.HttpNemoTeacher()
    assert http.transcribe(wav)["alignment"] == [{"word": "سلام"}]

    monkeypatch.setattr(
        "thesis_s2s.runtime.cascade.asr_http", lambda _payload: {"error": "service down"}
    )
    with pytest.raises(RuntimeError, match="service down"):
        http.transcribe(wav)


def test_teacher_selection_records_each_failed_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        batch_reasr,
        "HttpNemoTeacher",
        lambda: (_ for _ in ()).throw(RuntimeError("http unavailable")),
    )
    monkeypatch.setattr(
        batch_reasr,
        "FasterWhisperTeacher",
        lambda: (_ for _ in ()).throw(RuntimeError("faster unavailable")),
    )
    fallback = SimpleNamespace(model_name="whisper", transcribe=lambda _path: {})
    monkeypatch.setattr(batch_reasr, "WhisperTeacher", lambda: fallback)

    teacher, errors = batch_reasr._make_local_teacher(
        allow_whisper=True, prefer_http_nemo=True
    )
    assert teacher is fallback
    assert errors == {
        "http_nemo": "http unavailable",
        "faster_whisper": "faster unavailable",
    }
    assert batch_reasr._make_local_teacher(
        allow_whisper=False, prefer_http_nemo=False
    ) == (None, {})


def test_probe_and_transcribe_row_fail_closed(tmp_path: Path, monkeypatch) -> None:
    assert batch_reasr._probe_teacher(lambda _path: {"transcript_nemo": "سلام"}, tmp_path)
    assert not batch_reasr._probe_teacher(lambda _path: {}, tmp_path)
    assert not batch_reasr._probe_teacher(
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad")), tmp_path
    )

    teacher = SimpleNamespace(transcribe=lambda _path: {"transcript_nemo": "سلام"})
    extra, error = batch_reasr._transcribe_row(
        {"audio_filepath": str(tmp_path / "a.wav")}, None, None, None, teacher
    )
    assert extra == {"transcript_nemo": "سلام"}
    assert error is None

    teacher.transcribe = lambda _path: (_ for _ in ()).throw(RuntimeError("failure detail"))
    extra, error = batch_reasr._transcribe_row(
        {"audio_path": str(tmp_path / "b.wav")}, None, None, None, teacher
    )
    assert extra == {}
    assert error == "failure detail"


def test_run_manifest_dry_run_limit_and_resume(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    rows = [
        {"utt_id": "a", "audio_filepath": str(tmp_path / "a.wav")},
        {"utt_id": "b", "audio_filepath": str(tmp_path / "b.wav")},
    ]
    _write_rows(source, rows)
    monkeypatch.setattr(batch_reasr, "_load_speech_service", lambda: None)
    monkeypatch.setattr(batch_reasr, "_make_local_teacher", lambda **_kwargs: (None, {}))

    report = batch_reasr.run_manifest(source, output, limit=1, resume=False)
    written = _read_rows(output)
    assert report == {
        "n": 1,
        "ok": 0,
        "skipped": 1,
        "resumed": 0,
        "dry_run": True,
        "teacher": None,
        "workers": 1,
        "in_progress": False,
    }
    assert written[0]["reasr_note"] == "dry_run_speech_service_unavailable"
    assert "transcript_nemo" not in written[1]
    assert json.loads((tmp_path / "reasr_stats.json").read_text()) == report

    _write_rows(output, [{**rows[0], "transcript_nemo": "قبلی"}])
    report = batch_reasr.run_manifest(source, output, resume=True)
    written = _read_rows(output)
    assert report["resumed"] == 1
    assert report["ok"] == 1
    assert report["n"] == 1
    assert written[0]["transcript_nemo"] == "قبلی"
    assert written[1]["reasr_note"] == "dry_run_speech_service_unavailable"


def test_run_manifest_retries_failed_http_probe_and_parallelizes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    wav_a.touch()
    wav_b.touch()
    _write_rows(
        source,
        [
            {"utt_id": "a", "audio_filepath": str(wav_a)},
            {"utt_id": "b", "audio_filepath": str(wav_b)},
        ],
    )
    monkeypatch.setattr(batch_reasr, "_load_speech_service", lambda: None)

    bad = SimpleNamespace(
        model_name="http",
        transcribe=lambda _path: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    def transcribe(path: Path) -> dict:
        if path.name == "b.wav":
            raise RuntimeError("bad row")
        return {
            "transcript_nemo": "سلام",
            "language_status": "ok",
            "alignment": [],
            "teacher": "fallback",
        }

    good = SimpleNamespace(model_name="fallback", transcribe=transcribe)
    calls: list[bool] = []

    def select_teacher(*, allow_whisper: bool, prefer_http_nemo: bool):
        assert allow_whisper is True
        calls.append(prefer_http_nemo)
        return (bad, {}) if prefer_http_nemo else (good, {"http_nemo": "offline"})

    monkeypatch.setattr(batch_reasr, "_make_local_teacher", select_teacher)
    report = batch_reasr.run_manifest(
        source, output, resume=False, prefer_http_nemo=True, workers=2
    )
    written = _read_rows(output)

    assert calls == [True, False]
    assert report["teacher_probe_error"] == "http failed the input-audio probe"
    assert report["teacher_errors"] == {"http_nemo": "offline"}
    assert report["teacher"] == "fallback"
    assert report["workers"] == 2
    assert report["ok"] == 1
    assert report["skipped"] == 1
    assert written[0]["transcript_nemo"] == "سلام"
    assert written[0]["reasr_available"] is True
    assert written[1]["reasr_error"] == "bad row"


def test_run_manifest_uses_initialized_local_service_serially(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    wav = tmp_path / "sample.wav"
    wav.touch()
    _write_rows(source, [{"utt_id": "a", "audio_filepath": str(wav)}])

    initialized: list[bool] = []

    class SpeechService:
        def initialize(self) -> None:
            initialized.append(True)

    silero = ModuleType("silero_vad")
    monkeypatch.setitem(sys.modules, "silero_vad", silero)
    monkeypatch.setattr(
        batch_reasr,
        "_load_speech_service",
        lambda: (SpeechService, object(), object()),
    )
    monkeypatch.setattr(batch_reasr, "_probe_teacher", lambda _fn, _wav: True)
    monkeypatch.setattr(
        batch_reasr,
        "reasr_one",
        lambda *_args: {
            "transcript_nemo": "سلام",
            "language_status": "ok",
            "alignment": [],
            "teacher": "nemo-soroush",
        },
    )

    report = batch_reasr.run_manifest(
        source, output, resume=False, prefer_http_nemo=False, workers=8
    )
    assert initialized == [True]
    assert report["teacher"] == "nemo-soroush"
    assert report["workers"] == 1
    assert report["ok"] == 1
    assert _read_rows(output)[0]["reasr_available"] is True
