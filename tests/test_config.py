from pathlib import Path

import pytest

from thesis_s2s import config


def test_load_yaml_handles_empty_mapping_and_rejects_non_mapping(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("# intentionally empty\n", encoding="utf-8")
    assert config.load_yaml(empty) == {}

    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level YAML value must be a mapping"):
        config.load_yaml(sequence)


def test_portable_project_values_only_rewrites_project_local_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    inside = root / "data" / "sample.jsonl"
    outside = tmp_path / "external" / "sample.jsonl"
    payload = {
        "path": inside,
        "string": inside.as_posix(),
        "outside": outside,
        "nested": [inside, (inside.as_posix(), outside.as_posix())],
        "number": 7,
    }

    portable = config.portable_project_values(payload, root=root)

    assert portable["path"] == "data/sample.jsonl"
    assert portable["string"] == "data/sample.jsonl"
    assert portable["outside"] == outside.as_posix()
    assert portable["nested"] == [
        "data/sample.jsonl",
        ("data/sample.jsonl", outside.as_posix()),
    ]
    assert portable["number"] == 7


def test_config_path_uses_discovered_project_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "project_root", lambda: tmp_path)
    assert config.config_path("data.yaml") == tmp_path / "configs" / "data.yaml"
