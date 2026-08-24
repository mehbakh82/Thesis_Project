import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import write_wav
from thesis_s2s.data.moshi import export_moshi_finetune_dataset
from thesis_s2s.data.moshi_audit import audit_moshi_finetune_dataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authorized_pair(tmp_path: Path, split: str, index: int) -> dict:
    user = 0.2 * np.sin(2 * np.pi * 180 * np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE)
    response = 0.2 * np.sin(
        2 * np.pi * 240 * np.arange(SAMPLE_RATE // 2, dtype=np.float32) / SAMPLE_RATE
    )
    user_path = tmp_path / f"user-{index}.wav"
    response_path = tmp_path / f"response-{index}.wav"
    write_wav(user_path, user)
    write_wav(response_path, response)
    basis = "supervisor-approved-internal-research"
    return {
        "utt_id": f"pair-{index}",
        "session_id": f"session-{index}",
        "split": split,
        "audio_filepath": str(user_path),
        "response_audio_filepath": str(response_path),
        "response_text": "پاسخ آزمایشی",
        "source_span_start": 10.0,
        "source_user_interval": [10.0, 11.0],
        "source_response_interval": [11.2, 11.7],
        "license": "pending-youtube-rights-review",
        "license_verified": False,
        "internal_research_authorized": True,
        "authorization_basis": basis,
        "redistribution_allowed": False,
        "human_verified": index == 0,
        "rights_review": {
            "status": "approved",
            "authorization_basis": basis,
            "decision_complete": True,
            "permissions": {
                "internal_training": True,
                "thesis_reporting": True,
                "derived_artifacts": True,
                "redistribution": False,
            },
            "evidence_reference": "supervisor-decision",
            "approved_by": "thesis-supervisor",
            "approval_date": "2026-08-23",
        },
    }


def test_moshi_export_writes_stereo_response_training_schema(tmp_path: Path):
    source = tmp_path / "conversations.jsonl"
    rows = [
        _authorized_pair(tmp_path, "train", 0),
        _authorized_pair(tmp_path, "val", 1),
        _authorized_pair(tmp_path, "test", 2),
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "moshi"
    report = export_moshi_finetune_dataset(
        source,
        out_dir,
        report_path=tmp_path / "report.json",
        assistant_audio_mode="source",
    )

    assert report["schema"] == "kyutai_moshi_finetune_stereo_v1"
    assert report["exported_pairs"] == 3
    assert report["assistant_channel"] == 0
    assert report["user_channel"] == 1
    assert report["requirements"]["train_validation_test_present"] is True
    assert report["requirements"]["manual_verification_sample_present"] is True
    assert report["requirements"]["training_hours_100_to_200"] is False
    assert report["final_training_ready"] is False

    train_record = json.loads((out_dir / "train.jsonl").read_text().strip())
    wav_path = Path(train_record["path"])
    with wave.open(str(wav_path), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getframerate() == SAMPLE_RATE
        frames = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").reshape(-1, 2)
    assert np.max(np.abs(frames[:SAMPLE_RATE, 1])) > 0
    assert np.max(np.abs(frames[:SAMPLE_RATE, 0])) == 0
    assert np.max(np.abs(frames[int(1.2 * SAMPLE_RATE) :, 0])) > 0
    metadata = json.loads(wav_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert metadata["alignments"][0][0] == "پاسخ آزمایشی"
    assert metadata["alignments"][0][2] == "SPEAKER_MAIN"


    audit = audit_moshi_finetune_dataset(
        source,
        out_dir,
        tmp_path / "report.json",
        out_path=tmp_path / "audit.json",
        sample_csv_path=tmp_path / "sample.csv",
        sample_size=3,
    )
    assert audit["audit_passes"] is True
    assert audit["verified_export_pairs"] == 3
    assert audit["requirements"]["all_file_header_metadata_and_hash_checks_pass"] is True
    assert audit["sample"]["rows"] == 3
    assert audit["sample"]["human_review_complete"] is False
    assert (tmp_path / "sample.csv").read_text(encoding="utf-8-sig").count("\n") == 4


def test_moshi_audit_rejects_rehashed_swapped_channels(tmp_path: Path) -> None:
    source = tmp_path / "conversations.jsonl"
    rows = [
        _authorized_pair(tmp_path, "train", 0),
        _authorized_pair(tmp_path, "val", 1),
        _authorized_pair(tmp_path, "test", 2),
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "moshi"
    report_path = tmp_path / "report.json"
    export_moshi_finetune_dataset(
        source,
        out_dir,
        report_path=report_path,
        assistant_audio_mode="source",
    )

    manifest_path = out_dir / "train.jsonl"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    wav_path = Path(record["path"])
    with wave.open(str(wav_path), "rb") as handle:
        params = handle.getparams()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").reshape(-1, 2)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(pcm[:, ::-1].copy().tobytes())

    record["sha256"] = _sha256(wav_path)
    manifest_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    export_report = json.loads(report_path.read_text(encoding="utf-8"))
    export_report["manifest_sha256"]["train"] = _sha256(manifest_path)
    report_path.write_text(
        json.dumps(export_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    audit = audit_moshi_finetune_dataset(
        source,
        out_dir,
        report_path,
        out_path=tmp_path / "audit.json",
        sample_csv_path=tmp_path / "sample.csv",
        sample_size=3,
    )

    assert audit["audit_passes"] is False
    assert audit["requirements"]["actual_channel_order_and_user_audio_match"] is False
    assert audit["failure_counts"]["user_channel_content_mismatch"] == 1


def test_moshi_export_resumes_only_hashable_exact_outputs(tmp_path: Path, monkeypatch):
    source = tmp_path / "conversations.jsonl"
    rows = [
        _authorized_pair(tmp_path, "train", 0),
        _authorized_pair(tmp_path, "val", 1),
        _authorized_pair(tmp_path, "test", 2),
    ]
    source.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "moshi"
    first = export_moshi_finetune_dataset(
        source,
        out_dir,
        assistant_audio_mode="source",
    )
    assert first["requirements"]["immutable_file_hashes_present"] is True
    monkeypatch.setattr(
        "thesis_s2s.data.moshi.read_wav",
        lambda *_args, **_kwargs: pytest.fail("validated exports must not be decoded again"),
    )
    resumed = export_moshi_finetune_dataset(
        source,
        out_dir,
        assistant_audio_mode="source",
    )
    assert resumed["resume"] == {
        "validated_existing_pairs": 3,
        "newly_exported_pairs": 0,
        "invalid_existing_outputs_rebuilt": 0,
    }
    records = [
        json.loads(line)
        for name in ("train.jsonl", "val.jsonl", "test.jsonl")
        for line in (out_dir / name).read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["sha256"] and record["metadata_sha256"] for record in records)
    assert all(record["channels"] == 2 for record in records)
    assert all(record["sample_rate"] == SAMPLE_RATE for record in records)


def test_moshi_export_rejects_unauthorized_pairs(tmp_path: Path):
    row = _authorized_pair(tmp_path, "train", 0)
    row["internal_research_authorized"] = False
    source = tmp_path / "unauthorized.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="training authorization"):
        export_moshi_finetune_dataset(
            source, tmp_path / "must-not-exist", assistant_audio_mode="source"
        )


def test_moshi_export_validates_pair_limit(tmp_path: Path):
    source = tmp_path / "empty.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="max_pairs"):
        export_moshi_finetune_dataset(
            source, tmp_path / "out", max_pairs=0, assistant_audio_mode="source"
        )


def test_moshi_primary_export_uses_one_pinned_piper_voice(tmp_path: Path, monkeypatch):
    row = _authorized_pair(tmp_path, "train", 0)
    source = tmp_path / "piper.jsonl"
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    fake_model = tmp_path / "voice.onnx"
    fake_model.write_bytes(b"pinned-voice")
    monkeypatch.setattr(
        "thesis_s2s.data.moshi.MANA_PIPER_SHA256",
        hashlib.sha256(fake_model.read_bytes()).hexdigest(),
    )
    rendered = np.full(SAMPLE_RATE // 4, 0.1, dtype=np.float32)
    monkeypatch.setattr("thesis_s2s.runtime.tts.os_piper_model", lambda: fake_model)
    synthesis_options = []

    def render(_text, _sr, **options):
        synthesis_options.append(options)
        return rendered

    monkeypatch.setattr("thesis_s2s.runtime.tts.piper_synthesize", render)

    report = export_moshi_finetune_dataset(source, tmp_path / "moshi-piper")

    assert report["assistant_audio_mode"] == "piper"
    assert report["requirements"]["consistent_assistant_voice"] is True
    assert report["requirements"]["assistant_voice_model_pinned"] is True
    assert report["assistant_voice_model"]["sha256"]
    assert synthesis_options == [{"deterministic": True, "isolated": True}]
    record = json.loads((tmp_path / "moshi-piper" / "train.jsonl").read_text())
    metadata = json.loads(Path(record["path"]).with_suffix(".json").read_text())
    assert metadata["assistant_audio_mode"] == "piper"
    assert metadata["assistant_voice_model_sha256"]


def test_moshi_primary_export_rejects_unpinned_voice(tmp_path: Path, monkeypatch):
    wrong_model = tmp_path / "wrong.onnx"
    wrong_model.write_bytes(b"wrong")
    monkeypatch.setattr("thesis_s2s.runtime.tts.os_piper_model", lambda: wrong_model)
    source = tmp_path / "unused.jsonl"
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash does not match"):
        export_moshi_finetune_dataset(source, tmp_path / "must-not-exist")
