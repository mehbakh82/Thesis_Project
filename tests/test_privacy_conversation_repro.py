import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from thesis_s2s.audio import write_wav
from thesis_s2s.bargein.detector import DetectorConfig
from thesis_s2s.bargein.synthetic import _harmonic
from thesis_s2s.bargein.train import feature_rows_from_jsonl, train_feature_detector
from thesis_s2s.data.conversation import (
    audit_conversation_manifest,
    build_conversation_manifest,
    estimate_conversation_pair_yield,
    export_llama_omni2_questions,
    select_conversation_reserve_by_yield,
)
from thesis_s2s.data.diarize import _overlap_intervals, diarize_file
from thesis_s2s.data.qa_policy import load_qa_waiver
from thesis_s2s.repro import (
    _conversation_status,
    gpu_preflight,
    load_upstream_lock,
    release_snapshot,
    verify_upstream_lock,
)
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
    estimate = estimate_conversation_pair_yield(raw)
    build = build_conversation_manifest(raw, manifest, tmp_path / "clips")
    assert estimate["estimated_pairs"] == build["pairs"] == 2
    assert estimate["estimated_pair_hours"] == build["hours"]
    assert estimate["training_ready"] is False
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
    rejected = dict(row)
    rejected["manual_qa_reviewed"] = True
    rejected["human_verified"] = False
    raw.write_text(json.dumps(rejected, ensure_ascii=False) + "\n", encoding="utf-8")
    rejected_estimate = estimate_conversation_pair_yield(raw)
    rejected_build = build_conversation_manifest(
        raw, tmp_path / "rejected.jsonl", tmp_path / "rejected-clips"
    )
    assert rejected_estimate["estimated_pairs"] == rejected_build["pairs"] == 0
    assert rejected_estimate["skipped"] == {"manual_qa_rejected": 1}
    assert rejected_build["skipped"] == {"manual_qa_rejected": 1}
    rows[0]["internal_research_authorized"] = False
    unlicensed = tmp_path / "unlicensed.jsonl"
    unlicensed.write_text(
        "\n".join(json.dumps(item) for item in rows) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="training authorization"):
        export_llama_omni2_questions(unlicensed, tmp_path / "must-not-exist.json")


def test_reserve_selector_uses_minimum_whole_episode_prefix(tmp_path: Path):
    def row(episode_id: str, index: int) -> dict:
        return {
            "window_id": f"{episode_id}-window-{index}",
            "episode_id": episode_id,
            "channel": "Podcast",
            "automatic_multi_speaker_verified": True,
            "reference_alignment_status": "complete",
            "segments": [
                {"start": 0.0, "end": 10.0, "speaker": "A", "text": "پرسش"},
                {"start": 10.0, "end": 20.0, "speaker": "B", "text": "پاسخ"},
            ],
        }

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in rows) + "\n",
            encoding="utf-8",
        )

    primary = tmp_path / "primary.jsonl"
    reserve = tmp_path / "reserve.jsonl"
    primary_selection = tmp_path / "primary-selection.jsonl"
    reserve_selection = tmp_path / "reserve-selection.jsonl"
    write_jsonl(primary, [row("primary", 0)])
    reserve_rows = [row(f"reserve-{index}", index) for index in range(3)]
    write_jsonl(reserve, reserve_rows)
    write_jsonl(primary_selection, [{"episode_id": "primary", "csv_hours": 0.009}])
    write_jsonl(
        reserve_selection,
        [{"episode_id": f"reserve-{index}", "csv_hours": 0.009} for index in range(3)],
    )
    selected = tmp_path / "selected.jsonl"
    report = select_conversation_reserve_by_yield(
        primary,
        reserve,
        primary_selection,
        reserve_selection,
        selected,
        report_path=tmp_path / "selection-report.json",
        target_pair_hours=0.017,
        max_candidate_hours=0.028,
    )

    assert report["selected_reserve_episodes"] == 2
    assert report["selected_reserve_windows"] == 2
    assert report["target_reached"] is True
    assert report["candidate_cap_passes"] is True
    assert report["combined_candidate_hours"] == 0.027
    selected_rows = [json.loads(line) for line in selected.read_text().splitlines()]
    assert [item["episode_id"] for item in selected_rows] == ["reserve-0", "reserve-1"]
    assert (tmp_path / "selection-report.json").is_file()


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
    report_dir = tmp_path / "thesis-report"
    report_dir.mkdir()
    (report_dir / "thesis.tex").write_text("report source", encoding="utf-8")
    (report_dir / "thesis.pdf").write_bytes(b"%PDF-1.7\n")
    (report_dir / "thesis.aux").write_text("generated", encoding="utf-8")
    egg_info = tmp_path / "src" / "generated.egg-info"
    egg_info.mkdir()
    (egg_info / "PKG-INFO").write_text("generated", encoding="utf-8")
    (tmp_path / "README.md").write_text("snapshot", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("Apache License, Version 2.0", encoding="utf-8")
    recorded_proxy = tmp_path / "results" / "eval" / "interrupt_recorded_proxy.json"
    recorded_proxy.parent.mkdir(parents=True)
    recorded_proxy.write_text("{}", encoding="utf-8")
    cleanup = tmp_path / "results" / "hardware" / "storage_cleanup_20260831.json"
    cleanup.parent.mkdir(parents=True)
    cleanup.write_text("{}", encoding="utf-8")
    final_audit = tmp_path / "results" / "release" / "final_audit.json"
    final_audit.parent.mkdir(parents=True)
    final_audit.write_text("{}", encoding="utf-8")
    qwen35_smoke = tmp_path / "results" / "hardware" / "qwen35_local_smoke.json"
    qwen35_smoke.write_text("{}", encoding="utf-8")
    cascade_analysis = (
        tmp_path / "results" / "eval" / "cascade_validation_descriptive_analysis.json"
    )
    cascade_analysis.write_text("{}", encoding="utf-8")
    legacy_dir = tmp_path / "checkpoints" / "llama_omni2_fa"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "persian_omni2.pt").write_bytes(b"historical")
    (legacy_dir / "dummy_adapter.pt").write_bytes(b"tests only")
    monkeypatch.setattr("thesis_s2s.repro.project_root", lambda: tmp_path)
    monkeypatch.setattr("thesis_s2s.repro.gpu_inventory", lambda: {"device": "test"})
    report = release_snapshot(tmp_path / "snapshot.json")
    assert report["file_count"] >= 5
    assert report["files"]["README.md"]["sha256"]
    assert report["files"]["LICENSE"]["sha256"]
    assert "checkpoints/llama_omni2_fa/persian_omni2.pt" in report["files"]
    assert "checkpoints/llama_omni2_fa/dummy_adapter.pt" not in report["files"]
    assert "results/eval/interrupt_recorded_proxy.json" in report["files"]
    assert "results/hardware/storage_cleanup_20260831.json" in report["files"]
    assert "results/release/final_audit.json" in report["files"]
    assert "results/hardware/qwen35_local_smoke.json" in report["files"]
    assert "results/eval/cascade_validation_descriptive_analysis.json" in report["files"]
    assert "thesis-report/thesis.tex" in report["files"]
    assert "thesis-report/thesis.pdf" in report["files"]
    assert "thesis-report/thesis.aux" not in report["files"]
    assert "src/generated.egg-info/PKG-INFO" not in report["files"]
    assert (tmp_path / "snapshot.json").is_file()


def test_conversation_status_reports_staging_without_claiming_final_readiness(tmp_path: Path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "diarized_episode_audit_combined_authorized.json").write_text(
        json.dumps(
            {
                "windows": 1021,
                "episodes": 296,
                "automatic_multi_speaker_hours": 181.824,
                "reference_aligned_hours": 217.115,
                "requirements": {
                    "automatic_multi_speaker_hours_in_contract": True,
                    "reference_aligned_hours_in_contract": True,
                    "training_use_authorized": True,
                    "manual_qa_sample_present": False,
                },
            }
        ),
        encoding="utf-8",
    )
    (results / "conversation_yield_estimate_combined.json").write_text(
        json.dumps(
            {
                "estimated_pairs": 6017,
                "estimated_pair_hours": 105.727,
                "hours_100_to_200": True,
                "non_mutating_estimate": True,
            }
        ),
        encoding="utf-8",
    )

    status = _conversation_status(tmp_path)

    assert status["present"] is False
    assert status["thesis_coverage_ok"] is False
    assert status["staging"]["automatic_gates_pass"] is True
    assert status["staging"]["manual_qa_complete"] is False
    assert status["yield_estimate"]["hours"] == 105.727


def test_h100_training_readiness_is_independent_of_target_gpu(tmp_path: Path, monkeypatch):
    (tmp_path / "configs").mkdir()
    base_blob = tmp_path / "hf_cache" / "pinned" / "model.bin"
    base_blob.parent.mkdir(parents=True)
    base_blob.write_bytes(b"pinned base model")
    (tmp_path / "configs" / "moshika_7b_legacy.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "configs" / "moshi_h100.yaml").write_text(
        "moshi_paths:\n"
        "  hf_repo_id: kyutai/moshika-pytorch-bf16\n"
        "  moshi_path: hf_cache/pinned/model.bin\n"
        "  config_path: configs/moshika_7b_legacy.json\n",
        encoding="utf-8",
    )
    smoke_config = tmp_path / "configs" / "moshi_h100_smoke.yaml"
    smoke_config.write_text("max_steps: 1\n", encoding="utf-8")
    trainer = tmp_path / "third_party" / "checkouts" / "moshi-finetune" / "train.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("# pinned trainer\n", encoding="utf-8")
    launcher = tmp_path / "scripts" / "moshi_train_entry.py"
    launcher.parent.mkdir()
    launcher.write_text("# reviewed launcher\n", encoding="utf-8")
    lock = tmp_path / "third_party" / "UPSTREAMS.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "upstreams": [
                    {
                        "name": "Moshi-Finetune",
                        "repository": "https://example.invalid/moshi-finetune.git",
                        "revision": "b" * 40,
                        "license": "Apache-2.0",
                        "checkout_dir": "moshi-finetune",
                        "training_entrypoint": "torchrun -m train",
                    },
                    {
                        "name": "Moshika-PyTorch-BF16",
                        "repository": "https://example.invalid/moshika",
                        "revision": "d" * 40,
                        "license": "CC-BY-4.0",
                        "checkout_dir": "moshika",
                        "weights": {
                            "model.bin": {
                                "bytes": len(base_blob.read_bytes()),
                                "sha256": hashlib.sha256(base_blob.read_bytes()).hexdigest(),
                            }
                        },
                    },
                    {
                        "name": "Mana-Persian-Piper",
                        "repository": "https://example.invalid/mana-piper",
                        "revision": "c" * 40,
                        "license": "MIT",
                        "checkout_dir": "mana-piper",
                        "weights": {
                            "file": "voice.onnx",
                            "sha256": "PLACEHOLDER_VOICE_SHA256",
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    voice = tmp_path / "models" / "piper" / "voice.onnx"
    voice.parent.mkdir(parents=True)
    voice.write_bytes(b"pinned voice")
    lock.write_text(
        lock.read_text(encoding="utf-8").replace(
            "PLACEHOLDER_VOICE_SHA256", hashlib.sha256(voice.read_bytes()).hexdigest()
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    results.mkdir()
    (results / "conversation_audit.json").write_text(
        json.dumps({"thesis_coverage_ok": True, "pairs": 10, "hours": 100.0}),
        encoding="utf-8",
    )
    hardware = results / "hardware"
    hardware.mkdir()
    environment = hardware / "moshi_environment.json"
    environment.write_text(
        json.dumps(
            {
                "valid": True,
                "gpu": "NVIDIA H100 NVL",
                "torch_cuda": "12.4",
                "gates": {"isolated": True, "assets": True},
            }
        ),
        encoding="utf-8",
    )
    (hardware / "moshi_h100_smoke.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "wiring_gate_passes": True,
                "scientific_evidence": False,
                "hardware": {"peak_allocated_gb": 15.0},
                "artifacts": {
                    "environment_report_sha256": hashlib.sha256(
                        environment.read_bytes()
                    ).hexdigest(),
                    "source_config_sha256": hashlib.sha256(smoke_config.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    manifests = tmp_path / "data" / "processed" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "conversation_manual_qa.csv").write_text(
        "window_id,review_status,speaker_count_correct,speaker_assignment_correct,"
        "caption_acceptable,overlap_annotation_correct,reviewer_id\n"
        "w1,pass,yes,yes,yes,yes,R1\n",
        encoding="utf-8",
    )
    (manifests / "conversation_interruption_qa.csv").write_text(
        "candidate_id,review_status,speakers_distinct_correct,user_turn_boundary_correct,"
        "response_turn_boundary_correct,audible_overlap_correct,corrected_label,reviewer_id\n"
        "c1,pass,yes,yes,yes,yes,interrupt,R1\n",
        encoding="utf-8",
    )
    (results / "manual_qa_report.json").write_text(
        json.dumps(
            {
                "fail_closed": True,
                "counts": {"qa_decisions": 1, "incomplete": 0, "unknown_window_ids": 0},
            }
        ),
        encoding="utf-8",
    )
    (results / "interaction_qa_report.json").write_text(
        json.dumps(
            {
                "fail_closed": True,
                "qa_complete": True,
                "verified_interruption_present": True,
                "counts": {
                    "qa_decisions": 1,
                    "incomplete": 0,
                    "unknown_candidate_ids": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (results / "moshi_export_report.json").write_text(
        json.dumps(
            {
                "final_training_ready": True,
                "exported_pairs": 10,
                "exported_hours": 100.0,
                "requirements": {"coverage": True, "authorization": True},
            }
        ),
        encoding="utf-8",
    )
    (results / "moshi_export_audit.json").write_text(
        json.dumps(
            {
                "audit_passes": True,
                "training_ready_under_qa_waiver": True,
                "verified_export_pairs": 10,
                "verified_export_hours": 100.0,
                "requirements": {"files": True, "channels": True},
            }
        ),
        encoding="utf-8",
    )
    fake_cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
    fake_torch = SimpleNamespace(
        __version__="2.6.0",
        version=SimpleNamespace(cuda="12.4"),
        cuda=fake_cuda,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("thesis_s2s.repro.project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "thesis_s2s.repro.verify_upstream_lock",
        lambda **_: {"schema_version": 1, "valid": True, "checks": []},
    )
    monkeypatch.setattr(
        "thesis_s2s.repro.gpu_inventory",
        lambda: {
            "device": "NVIDIA H100 NVL",
            "total_gb": 93.09,
            "official_size": False,
        },
    )
    monkeypatch.setattr("thesis_s2s.repro._nvidia_smi", lambda: {"available": True})

    report = gpu_preflight()

    assert report["training_hardware_ready"] is True
    assert report["evaluation_hardware_ready"] is False
    assert report["adaptation_run_ready"] is True
    assert report["adaptation_launch_safe_now"] is False
    assert report["trainer_stack_ready"] is True
    assert report["training_data_ready"] is True
    assert report["training_gates"]["reviewed_training_entrypoint_available"] is True
    assert report["training_gates"]["one_step_official_wiring_smoke_passes"] is True
    assert report["training_gates"]["base_model_files_pinned"] is True
    assert report["training_gates"]["assistant_voice_target_pinned"] is True
    assert report["evaluation_gates"]["physical_gpu_12_to_24_gb"] is False
    assert report["training_data_policy"] == "strict"
    assert report["strict_training_data_ready"] is True
    assert report["training_data_ready_under_qa_waiver"] is False

    waiver_path = tmp_path / "configs" / "conversation_qa_waiver.yaml"
    waiver_path.write_text(
        """schema_version: 1
status: acknowledged
policy: automatic_only_documented_waiver
decision_date: "2026-08-24"
decision_authority: student_project_owner
supervisor_approval_claimed: false
reason: insufficient_time_and_no_available_delegate
scope:
  window_manual_qa: waived
  interaction_manual_qa: waived
  internal_training: allowed_under_waiver
claims:
  human_verified_data: false
  human_verified_interruptions: false
  strict_thesis_data_coverage: false
preservation:
  qa_sheets: true
  reviewer_guides: true
  playback_helper: true
  future_review_supported: true
""",
        encoding="utf-8",
    )
    waiver = load_qa_waiver(waiver_path)
    claims = {
        "human_verified_data": False,
        "human_verified_interruptions": False,
        "strict_thesis_data_coverage": False,
    }
    (results / "conversation_audit.json").write_text(
        json.dumps(
            {
                "thesis_coverage_ok": False,
                "training_ready_under_qa_waiver": True,
                "qa_policy": waiver,
                "claims": claims,
                "pairs": 10,
                "hours": 100.0,
            }
        ),
        encoding="utf-8",
    )
    (results / "moshi_export_report.json").write_text(
        json.dumps(
            {
                "final_training_ready": False,
                "training_ready_under_qa_waiver": True,
                "requirements": {"manual_verification": False},
                "waiver_requirements": {"automatic_coverage": True},
                "qa_policy": waiver,
                "claims": claims,
                "exported_pairs": 10,
                "exported_hours": 100.0,
            }
        ),
        encoding="utf-8",
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("MANUAL_QA_FA.md", "INTERRUPTION_QA_FA.md"):
        (docs / name).write_text("preserved\n", encoding="utf-8")
    (tmp_path / "scripts" / "review_interaction_candidate.py").write_text(
        "# preserved helper\n", encoding="utf-8"
    )

    waiver_report = gpu_preflight()

    assert waiver_report["qa_waiver"]["selected_for_training"] is True
    assert waiver_report["training_data_policy"] == "automatic_only_documented_waiver"
    assert waiver_report["training_data_ready_under_qa_waiver"] is True
    assert waiver_report["training_data_ready"] is True
    assert waiver_report["strict_training_data_ready"] is False
    assert waiver_report["adaptation_run_ready"] is True
