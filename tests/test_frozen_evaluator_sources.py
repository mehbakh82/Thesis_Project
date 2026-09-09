import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_frozen_evaluator_sources import verify_manifest


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path):
    current = b"portable source\n"
    historical = b"frozen source\n"
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "scripts/evaluator.py").write_bytes(current)
    (tmp_path / "results/plan.json").write_text(
        json.dumps({"binding": {"sha256": _digest(historical)}}), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "historical_commit": "a" * 40,
        "entries": [
            {
                "path": "scripts/evaluator.py",
                "current_sha256": _digest(current),
                "historical_sha256": _digest(historical),
                "bindings": [["results/plan.json", "binding.sha256"]],
            }
        ],
    }
    return manifest, historical


def test_verifies_current_historical_and_receipt_binding(tmp_path: Path):
    manifest, historical = _fixture(tmp_path)
    report = verify_manifest(manifest, tmp_path, lambda _commit, _path: historical)
    assert report["status"] == "passed"
    assert report["entries_verified"] == 1
    assert report["receipt_bindings_verified"] == 1


def test_rejects_current_source_drift(tmp_path: Path):
    manifest, historical = _fixture(tmp_path)
    (tmp_path / "scripts/evaluator.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="current source drift"):
        verify_manifest(manifest, tmp_path, lambda _commit, _path: historical)
