import json
from pathlib import Path

import pytest

from thesis_s2s.data import factory


def _jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_hours_from_jsonl_fails_closed_on_malformed_or_invalid_rows(tmp_path):
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"duration": 1}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"malformed\.jsonl:2"):
        factory._hours_from_jsonl(malformed)

    non_object = tmp_path / "list.jsonl"
    _jsonl(non_object, [[]])
    with pytest.raises(ValueError, match="expected JSON object"):
        factory._hours_from_jsonl(non_object)

    invalid_duration = tmp_path / "duration.jsonl"
    _jsonl(invalid_duration, [{"duration": -1}])
    with pytest.raises(ValueError, match="finite and non-negative"):
        factory._hours_from_jsonl(invalid_duration)

    valid = tmp_path / "valid.jsonl"
    _jsonl(valid, [{"duration": 1800, "text": "caption"}, {"duration": 900}])
    assert factory._hours_from_jsonl(valid) == (2, 0.75, 1)


def test_load_report_fails_closed_on_invalid_json_or_shape(tmp_path):
    assert factory._load_report(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON report"):
        factory._load_report(malformed)
    malformed.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        factory._load_report(malformed)


def test_promote_requires_replacement_before_archiving(tmp_path, monkeypatch):
    manifests = tmp_path / "data" / "processed" / "manifests"
    manifests.mkdir(parents=True)
    filtered = manifests / "filtered.jsonl"
    filtered.write_text("old", encoding="utf-8")
    monkeypatch.setattr(factory, "project_root", lambda: tmp_path)

    report = factory.promote_caption_training_mix()

    assert report["copied"] is False
    assert filtered.read_text(encoding="utf-8") == "old"
    assert not (manifests / "filtered_nemo_teacher.jsonl").exists()


def test_promote_archives_once_and_atomically_replaces(tmp_path, monkeypatch):
    manifests = tmp_path / "data" / "processed" / "manifests"
    manifests.mkdir(parents=True)
    filtered = manifests / "filtered.jsonl"
    caption = manifests / "filtered_caption.jsonl"
    filtered.write_text("old", encoding="utf-8")
    caption.write_text("new", encoding="utf-8")
    monkeypatch.setattr(factory, "project_root", lambda: tmp_path)

    report = factory.promote_caption_training_mix()

    assert report["copied"] is True
    assert report["archived_nemo_mix"] is True
    assert filtered.read_text(encoding="utf-8") == "new"
    assert (manifests / "filtered_nemo_teacher.jsonl").read_text(encoding="utf-8") == "old"
    assert not list(manifests.glob("*.tmp"))


def test_atomic_copy_preserves_destination_and_cleans_temp_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_text("new", encoding="utf-8")
    destination.write_text("old", encoding="utf-8")
    monkeypatch.setattr(factory.shutil, "copy2", lambda *_: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError, match="boom"):
        factory._copy_atomic(source, destination)

    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.tmp"))


def test_scale_corpus_never_diarizes_stale_filtered_output(tmp_path, monkeypatch):
    manifests = tmp_path / "data" / "processed" / "manifests"
    manifests.mkdir(parents=True)
    _jsonl(manifests / "filtered.jsonl", [{"duration": 3600, "text": "stale"}])
    (tmp_path / "results").mkdir()
    (tmp_path / "docs").mkdir()
    monkeypatch.setattr(factory, "project_root", lambda: tmp_path)
    monkeypatch.setattr(
        factory.SessionStore, "export_manifest", lambda _self: {"hours": 0.0, "n": 0}
    )
    monkeypatch.setattr(
        factory,
        "annotate_manifest",
        lambda *args, **kwargs: pytest.fail("stale manifest was diarized"),
    )

    report = factory.scale_corpus(caption_in=tmp_path / "missing.jsonl")
    snapshot = json.loads((tmp_path / "results" / "dataset_card_snapshot.json").read_text())

    assert report["filtered_current_run"] is False
    assert report["diarization"]["attempted"] == 0
    assert snapshot["filtered_caption_hours"] == 0.0
    assert snapshot["filtered_caption_utts"] == 0
    assert snapshot["corpus_audit"]["reason"] == "filtered_manifest_missing"


def test_scale_corpus_diarizes_only_the_current_atomic_copy(tmp_path, monkeypatch):
    manifests = tmp_path / "data" / "processed" / "manifests"
    manifests.mkdir(parents=True)
    source = manifests / "youtube_all.jsonl"
    source.write_text("source", encoding="utf-8")
    (manifests / "filtered.jsonl").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(factory, "project_root", lambda: tmp_path)

    def filter_current(_source, output, **_kwargs):
        output.write_text("current", encoding="utf-8")
        return {"n": 1, "hours": 0.5, "min_hours_ok": True}

    observed = {}

    def annotate_current(path, _output, *, limit):
        observed["content"] = path.read_text(encoding="utf-8")
        observed["limit"] = limit
        return {"attempted": 1, "ok": 1}

    monkeypatch.setattr(factory, "filter_hours", filter_current)
    monkeypatch.setattr(factory, "annotate_manifest", annotate_current)
    monkeypatch.setattr(
        factory.SessionStore, "export_manifest", lambda _self: {"hours": 0.0, "n": 0}
    )
    monkeypatch.setattr(factory, "refresh_dataset_card", lambda report: observed.setdefault("report", report))

    report = factory.scale_corpus(diarize_limit=7)

    assert report["filtered_current_run"] is True
    assert observed["content"] == "current"
    assert observed["limit"] == 7
    assert observed["report"] is report


def test_run_factory_marks_stale_filtered_output_as_not_current(tmp_path, monkeypatch):
    manifests = tmp_path / "data" / "processed" / "manifests"
    manifests.mkdir(parents=True)
    (manifests / "filtered.jsonl").write_text("stale", encoding="utf-8")
    (tmp_path / "results").mkdir()
    observed = {}
    monkeypatch.setattr(factory, "project_root", lambda: tmp_path)
    monkeypatch.setattr(factory, "run_ingest", lambda **kwargs: {"ok": True, **kwargs})
    monkeypatch.setattr(
        factory,
        "filter_hours",
        lambda *args, **kwargs: pytest.fail("missing input was filtered"),
    )
    monkeypatch.setattr(
        factory,
        "write_synthetic_duplex",
        lambda *args, **kwargs: {"n": 4, "hours": 0.01},
    )
    monkeypatch.setattr(
        factory.SessionStore, "export_manifest", lambda _self: {"hours": 0.0, "n": 0}
    )
    monkeypatch.setattr(factory, "refresh_dataset_card", lambda report: observed.setdefault("report", report))

    report = factory.run_factory(max_audio_hours=2.0, max_episodes=3)

    assert report["filtered_current_run"] is False
    assert report["ingest"]["max_hours"] == 2.0
    assert report["ingest"]["max_episodes"] == 3
    assert observed["report"] is report
