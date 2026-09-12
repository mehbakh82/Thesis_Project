import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.runtime import session_log
from thesis_s2s.runtime.session_log import SessionMeta, SessionStore, _append_jsonl_atomic


def _turn(store: SessionStore, meta: SessionMeta):
    return store.add_turn(
        meta,
        prompt_id="warmup_time",
        interrupt_label="none",
        user_audio=np.ones(160, dtype=np.float32) * 0.1,
        t_first_audio_ms=100.0,
        t_barge_in_ms=None,
        stopped=False,
    )


def test_session_metadata_and_turn_numeric_inputs_fail_closed(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError, match="vram_gb"):
        store.start(SessionMeta("S1", "P1", "under_60", True, vram_gb=float("nan")))
    with pytest.raises(ValueError, match="memory_capped"):
        store.start(SessionMeta("S1", "P1", "under_60", True, memory_capped="no"))

    meta = SessionMeta("S2", "P2", "under_60", True, retention="metrics")
    with pytest.raises(ValueError, match="t_first_audio_ms"):
        store.add_turn(
            meta,
            prompt_id="warmup_time",
            interrupt_label="none",
            user_audio=np.ones(160, dtype=np.float32),
            t_first_audio_ms=float("inf"),
            t_barge_in_ms=None,
            stopped=False,
        )
    with pytest.raises(ValueError, match="stopped"):
        store.add_turn(
            meta,
            prompt_id="warmup_time",
            interrupt_label="none",
            user_audio=np.ones(160, dtype=np.float32),
            t_first_audio_ms=None,
            t_barge_in_ms=None,
            stopped=1,
        )
    with pytest.raises(ValueError, match="finite"):
        store.add_turn(
            meta,
            prompt_id="warmup_time",
            interrupt_label="none",
            user_audio=np.asarray([float("nan")], dtype=np.float32),
            t_first_audio_ms=None,
            t_barge_in_ms=None,
            stopped=False,
        )


def test_atomic_jsonl_append_preserves_existing_file_on_failure(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "rows.jsonl"
    _append_jsonl_atomic(path, {"index": 0})
    old_bytes = path.read_bytes()
    monkeypatch.setattr(
        session_log.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        _append_jsonl_atomic(path, {"index": 1})
    assert path.read_bytes() == old_bytes
    assert list(tmp_path.glob(".rows.jsonl.*.tmp")) == []

    monkeypatch.undo()
    path.write_bytes(b'{"index": 0}')
    with pytest.raises(ValueError, match="truncated JSONL"):
        _append_jsonl_atomic(path, {"index": 1})


def test_atomic_jsonl_append_serializes_concurrent_writers(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: _append_jsonl_atomic(path, lambda n: {"index": n}), range(32)))
    persisted = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["index"] for row in rows) == list(range(32))
    assert [row["index"] for row in persisted] == list(range(32))


def test_existing_session_rejects_changed_hardware_provenance(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    original = SessionMeta(
        "S1",
        "P1",
        "under_60",
        True,
        gpu_name="NVIDIA GeForce RTX 4090",
        vram_gb=24.0,
        memory_capped=False,
        retention="metrics",
    )
    store.start(original)
    changed = SessionMeta(
        "S1",
        "P1",
        "under_60",
        True,
        gpu_name="NVIDIA H100 NVL",
        vram_gb=24.0,
        memory_capped=True,
        retention="metrics",
    )
    with pytest.raises(ValueError, match="hardware provenance"):
        store.start(changed)


def test_turn_append_failure_removes_orphan_audio(tmp_path: Path, monkeypatch) -> None:
    store = SessionStore(tmp_path)
    meta = SessionMeta("S1", "P1", "under_60", True, retention="audio")
    store.start(meta)
    monkeypatch.setattr(
        session_log.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("append failed")),
    )
    with pytest.raises(OSError, match="append failed"):
        _turn(store, meta)
    folder = store.session_dir("S1")
    assert not list(folder.glob("*.wav"))
    assert (folder / "turns.jsonl").read_bytes() == b""


def test_manifest_export_is_atomic_and_preserves_old_file(tmp_path: Path, monkeypatch) -> None:
    store = SessionStore(tmp_path / "recordings")
    meta = SessionMeta("S1", "P1", "under_60", True, retention="metrics")
    _turn(store, meta)
    output = tmp_path / "manifest.jsonl"
    output.write_text('{"old": true}\n', encoding="utf-8")
    old_bytes = output.read_bytes()
    monkeypatch.setattr(
        session_log.os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("export failed")),
    )
    with pytest.raises(OSError, match="export failed"):
        store.export_manifest(output)
    assert output.read_bytes() == old_bytes
    assert list(tmp_path.glob(".manifest.jsonl.*.tmp")) == []


def test_official_study_hardware_requires_explicit_uncapped_provenance(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    folder = store.session_dir("legacy")
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "session_id": "legacy",
                "speaker_id": "P1",
                "age_bin": "under_60",
                "consent": True,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "vram_gb": 24.0,
                "retention": "metrics",
            }
        ),
        encoding="utf-8",
    )
    (folder / "turns.jsonl").write_text(
        json.dumps(
            {
                "retention": "metrics",
                "interrupt_label": "none",
                "stopped": False,
                "t_first_audio_ms": 100.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = store.study_summary()
    assert report["requirements"]["physical_gpu_12_to_24_gb"] is False
    assert report["official_t_first_audio_p50_ms"] is None
