import json
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.audio import write_wav
from thesis_s2s.bargein.detector import DetectorConfig
from thesis_s2s.bargein.synthetic import _harmonic
from thesis_s2s.bargein.train import feature_rows_from_jsonl, train_feature_detector
from thesis_s2s.data.conversation import (
    audit_conversation_manifest,
    build_conversation_manifest,
    export_llama_omni2_questions,
)
from thesis_s2s.data.diarize import _overlap_intervals, diarize_file
from thesis_s2s.repro import load_upstream_lock, release_snapshot, verify_upstream_lock
from thesis_s2s.runtime.session_log import SessionMeta, SessionStore


def test_feature_and_metrics_retention_write_no_wav(tmp_path: Path):
    interaction = _harmonic(0.45, 190)
    user = _harmonic(0.3, 170)

    feature_store = SessionStore(tmp_path / "features")
    feature_meta = SessionMeta("SF", "PF", "under_60", True, retention="features")
    feature_rec = feature_store.add_turn(
        feature_meta,
        prompt_id="interrupt_story",
        interrupt_label="interrupt",
        user_audio=user,
        interaction_audio=interaction,
        t_first_audio_ms=120.0,
        t_barge_in_ms=50.0,
        stopped=True,
    )
    assert feature_rec.audio_filepath == ""
    assert feature_rec.interaction_audio_filepath == ""
    assert len(feature_rec.privacy_feature_vector) == 90
    assert not list((tmp_path / "features").rglob("*.wav"))

    metrics_store = SessionStore(tmp_path / "metrics")
    metrics_meta = SessionMeta("SM", "PM", "under_60", True, retention="metrics")
    metrics_rec = metrics_store.add_turn(
        metrics_meta,
        prompt_id="warmup_time",
        interrupt_label="none",
        user_audio=user,
        interaction_audio=interaction,
        t_first_audio_ms=150.0,
        t_barge_in_ms=None,
        stopped=False,
    )
    assert metrics_rec.privacy_feature_vector == []
    assert not list((tmp_path / "metrics").rglob("*.wav"))


def test_feature_detector_is_group_heldout(tmp_path: Path):
    manifest = tmp_path / "features.jsonl"
    rows = []
    for group_index in range(6):
        for label, centre in (("none", -45.0), ("interrupt", -10.0)):
            vector = np.full(90, centre, dtype=np.float32)
            vector[1] = 0.1
            rows.append(
                {
                    "speaker_id": f"P{group_index}",
                    "interrupt_label": label,
                    "privacy_feature_vector": vector.tolist(),
                }
            )
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    loaded = feature_rows_from_jsonl(manifest)
    report = train_feature_detector(
        loaded,
        seed=3,
        out_dir=tmp_path / "out",
        detector_config=DetectorConfig(n_estimators=10, max_depth=2, min_samples_leaf=1),
    )
    assert report["evidence_class"] == "recorded_features_heldout"
    assert report["group_overlap"] is False
    assert report["proposed"]["n"] == 2
    assert report["proposed_ci95"]["accuracy"] is not None
    assert (tmp_path / "out" / "feature_heldout_report.json").is_file()


def test_conversation_builder_pairs_and_exports(tmp_path: Path):
    source = tmp_path / "conversation.wav"
    write_wav(source, _harmonic(4.0, 180))
    raw = tmp_path / "diarized.jsonl"
    row = {
        "session_id": "natural-session-1",
        "audio_filepath": str(source),
        "license": "test-license",
        "license_verified": False,
        "human_verified": True,
        "segments": [
            {"start": 0.0, "end": 0.9, "speaker": "A", "text": "سلام حال شما چطور است"},
            {"start": 0.75, "end": 1.05, "speaker": "B", "text": "بله"},
            {"start": 1.5, "end": 2.25, "speaker": "A", "text": "امروز هوا چطور است"},
            {"start": 2.4, "end": 3.35, "speaker": "B", "text": "امروز هوا خوب است"},
        ],
    }
    raw.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = tmp_path / "conversations.jsonl"
    with pytest.raises(PermissionError, match="training authorization"):
        build_conversation_manifest(raw, manifest, tmp_path / "clips")
    assert not manifest.exists()
    row.update(
        {
            "internal_research_authorized": True,
            "authorization_basis": "supervisor-approved-internal-research",
            "redistribution_allowed": False,
            "rights_review": {
                "scope_type": "channel",
                "scope_id": "test-source",
                "status": "approved",
                "authorization_basis": "supervisor-approved-internal-research",
                "license_name": "",
                "decision_complete": True,
                "permissions": {
                    "internal_training": True,
                    "thesis_reporting": True,
                    "derived_artifacts": True,
                    "redistribution": False,
                },
                "evidence_reference": "supervisor-approval",
                "approved_by": "thesis-supervisor",
                "approval_date": "2026-08-23",
                "notes": "",
            },
        }
    )
    raw.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    build = build_conversation_manifest(raw, manifest, tmp_path / "clips")
    assert build["pairs"] == 2
    assert build["label_counts"]["backchannel"] == 1
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len({item["split"] for item in rows}) == 1
    assert all(Path(item["response_audio_filepath"]).is_file() for item in rows)

    audit = audit_conversation_manifest(manifest, tmp_path / "audit.json")
    assert audit["requirements"]["response_pairs_present"] is True
    assert audit["requirements"]["manual_verification_sample_present"] is True
    assert audit["requirements"]["hours_100_to_200"] is False

    exported = export_llama_omni2_questions(manifest, tmp_path / "questions.json")
    assert exported["conversations"] == 2
    rows[0]["internal_research_authorized"] = False
    unlicensed = tmp_path / "unlicensed.jsonl"
    unlicensed.write_text(
        "\n".join(json.dumps(item) for item in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="training authorization"):
        export_llama_omni2_questions(unlicensed, tmp_path / "must-not-exist.json")


def test_diarization_contract_uses_float32_and_derives_overlap(tmp_path: Path, monkeypatch):
    source = tmp_path / "sample.wav"
    write_wav(source, _harmonic(0.2, 200))
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "turns": [
                        {"start": 0.0, "end": 1.0, "speaker": "A"},
                        {"start": 0.6, "end": 1.2, "speaker": "B"},
                    ],
                    "exclusive_turns": [],
                }
            ).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("thesis_s2s.data.diarize.urlopen", fake_urlopen)
    report = diarize_file(source, "http://diarizer")
    assert captured["url"].endswith("/diarize?sample_rate=16000")
    assert len(captured["data"]) == 3200 * 4
    assert report["overlap_intervals"] == [[0.6, 1.0]]
    assert report["speaker_turns"][0]["speaker"] == "A"
    assert _overlap_intervals(report["speaker_turns"]) == [[0.6, 1.0]]


def test_upstream_lock_and_release_snapshot(tmp_path: Path, monkeypatch):
    lock = tmp_path / "third_party" / "UPSTREAMS.lock.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "upstreams": [
                    {
                        "name": "example",
                        "repository": "https://example.invalid/repo.git",
                        "revision": "a" * 40,
                        "license": "test",
                        "checkout_dir": "example",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_upstream_lock(lock)["upstreams"][0]["revision"] == "a" * 40
    assert verify_upstream_lock(lock)["valid"] is True

    for name in ("configs", "docs", "src", "tests"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "evidence.txt").write_text(name, encoding="utf-8")
    egg_info = tmp_path / "src" / "generated.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated", encoding="utf-8")
    (tmp_path / "README.md").write_text("snapshot", encoding="utf-8")
    monkeypatch.setattr("thesis_s2s.repro.project_root", lambda: tmp_path)
    monkeypatch.setattr("thesis_s2s.repro.gpu_inventory", lambda: {"device": "test"})
    report = release_snapshot(tmp_path / "snapshot.json")
    assert report["file_count"] >= 5
    assert report["files"]["README.md"]["sha256"]
    assert "src/generated.egg-info/PKG-INFO" not in report["files"]
    assert (tmp_path / "snapshot.json").is_file()
