import json
from pathlib import Path

import numpy as np
import pytest
import torch

from thesis_s2s.audio import to_float32_mono
from thesis_s2s.config import load_yaml
from thesis_s2s.data.audit import audit_caption_alignment, audit_manifest
from thesis_s2s.model.llama_omni2 import checkpoint_runtime_status
from thesis_s2s.runtime.session_log import SessionMeta, SessionStore


def test_integer_pcm_normalization():
    pcm = np.array([-32768, 0, 32767], dtype=np.int16)
    audio = to_float32_mono(pcm)
    assert audio.dtype == np.float32
    assert audio[0] == pytest.approx(-1.0)
    assert audio[-1] == pytest.approx(32767 / 32768)


def test_all_project_yaml_is_nested_correctly():
    root = Path(__file__).parents[1]
    configs = {path.name: load_yaml(path) for path in (root / "configs").glob("*.yaml")}
    assert isinstance(configs["bargein.yaml"]["gbdt"], dict)
    assert "verbatim_version" in configs["data.yaml"]
    assert "t_barge_in" in configs["metrics.yaml"]


def test_legacy_checkpoint_is_fail_closed(tmp_path: Path):
    path = tmp_path / "legacy.pt"
    torch.save({"proj": {}}, path)
    status = checkpoint_runtime_status(path)
    assert status["exists"] is True
    assert status["runtime_ready"] is False
    assert status["artifact_kind"] == "legacy_untyped"


def test_session_requires_safe_ids_consent_and_preserves_turns(tmp_path: Path):
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.session_dir("../escape")
    with pytest.raises(PermissionError):
        store.start(SessionMeta("S1", "P1", "under_60", consent=False))
    meta = SessionMeta("S1", "P1", "under_60", consent=True)
    store.start(meta)
    turns = store.session_dir("S1") / "turns.jsonl"
    turns.write_text('{"utt_id":"existing"}\n', encoding="utf-8")
    store.start(meta)
    assert "existing" in turns.read_text(encoding="utf-8")


def test_manifest_audit_detects_group_leakage(tmp_path: Path):
    manifest = tmp_path / "data.jsonl"
    rows = [
        {
            "utt_id": "a",
            "duration": 2,
            "split": "train",
            "source_csv": "same.csv",
            "interrupt_label": "none",
            "text": "سلام",
        },
        {
            "utt_id": "b",
            "duration": 2,
            "split": "test",
            "source_csv": "same.csv",
            "interrupt_label": "interrupt",
            "text": "سلام",
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    report = audit_manifest(manifest, check_files=False)
    assert report["group_split_leaks"] == 1
    assert report["text_cross_split_duplicates"] == 1
    assert report["integrity_ok"] is False
    assert report["thesis_coverage_ok"] is False


def test_alignment_audit_flags_low_similarity_and_repetition(tmp_path: Path):
    manifest = tmp_path / "paired.jsonl"
    rows = [
        {
            "utt_id": "good",
            "duration": 2,
            "channel": "A",
            "transcript_caption": "سلام حال شما خوب است",
            "transcript_nemo": "سلام حال شما خوب است",
            "teacher": "test",
        },
        {
            "utt_id": "bad",
            "duration": 3,
            "channel": "A",
            "transcript_caption": "این جمله با صدای واقعی تفاوت کامل دارد",
            "transcript_nemo": " ".join(["از"] * 20),
            "teacher": "test",
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    report = audit_caption_alignment(manifest)

    assert report["paired_rows"] == 2
    assert report["alignment_risk"] is True
    assert report["repetitive_teacher_rows"] == 1
    assert report["token_sequence_similarity"]["low_lt_0_4_fraction"] == 0.5
