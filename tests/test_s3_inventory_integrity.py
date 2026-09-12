import json
import subprocess

import pytest

from thesis_s2s.data import s3_inventory


def _credentials(monkeypatch) -> None:
    monkeypatch.delenv("THESIS_RCLONE_REMOTE", raising=False)
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "access-test")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "secret-test")
    monkeypatch.setenv("S3_ENDPOINT", "https://s3.example.invalid")
    monkeypatch.setenv("S3_BUCKET", "asr")
    monkeypatch.setenv("S3_PROVIDER", "Minio")


def test_rclone_configuration_rejects_secret_bearing_or_ambiguous_values(monkeypatch) -> None:
    _credentials(monkeypatch)
    monkeypatch.setenv("S3_ENDPOINT", "https://user:secret@s3.example.invalid")
    with pytest.raises(ValueError, match="without credentials"):
        s3_inventory.rclone_env()

    _credentials(monkeypatch)
    monkeypatch.setenv("S3_BUCKET", "asr/path")
    with pytest.raises(ValueError, match="simple bucket"):
        s3_inventory.rclone_env()

    _credentials(monkeypatch)
    monkeypatch.setenv("S3_PROVIDER", "Minio;unsafe")
    with pytest.raises(ValueError, match="simple provider"):
        s3_inventory.rclone_env()


def test_rclone_timeout_must_be_positive_and_finite(monkeypatch) -> None:
    monkeypatch.delenv("RCLONE_TIMEOUT_SECONDS", raising=False)
    assert s3_inventory.rclone_timeout_seconds() == 3600.0
    for value in ("0", "-1", "nan", "inf", "invalid"):
        monkeypatch.setenv("RCLONE_TIMEOUT_SECONDS", value)
        with pytest.raises(ValueError, match="positive finite"):
            s3_inventory.rclone_timeout_seconds()


def test_listing_uses_validated_timeout(monkeypatch) -> None:
    monkeypatch.setattr(s3_inventory, "rclone_prefix", lambda: ["rclone"])
    monkeypatch.setattr(s3_inventory, "rclone_path", lambda value: value)
    monkeypatch.setattr(s3_inventory, "rclone_process_env", lambda: {"SAFE": "1"})
    monkeypatch.setattr(s3_inventory, "rclone_timeout_seconds", lambda: 12.5)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, stdout="folder\n")

    monkeypatch.setattr(s3_inventory.subprocess, "run", fake_run)
    assert s3_inventory.lsd(":s3:asr") == "folder\n"
    assert calls[0][1]["timeout"] == 12.5


def test_inventory_failure_is_redacted_persisted_and_nonzero(tmp_path, monkeypatch) -> None:
    _credentials(monkeypatch)
    secret = "do-not-persist-this-secret"

    def failed_listing(path: str) -> str:
        if path.endswith("asr"):
            return "top-level\n"
        raise subprocess.CalledProcessError(7, ["rclone"], stderr=secret)

    monkeypatch.setattr(s3_inventory, "lsd", failed_listing)
    output = tmp_path / "inventory.json"
    with pytest.raises(RuntimeError, match="inventory incomplete"):
        s3_inventory.inventory(output, [":s3:asr/restricted"])

    raw = output.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert secret not in raw
    assert payload["status"] == "failed"
    assert payload["failures"]["extra::s3:asr/restricted"]["returncode"] == 7
    assert list(tmp_path.glob(".inventory.json.*.tmp")) == []


def test_successful_inventory_is_status_labelled_and_atomic(tmp_path, monkeypatch) -> None:
    _credentials(monkeypatch)
    monkeypatch.setattr(s3_inventory, "lsd", lambda path: f"{path}/entry\n")
    output = tmp_path / "inventory.json"

    payload = s3_inventory.inventory(output, [":s3:asr/extra"])

    assert payload["status"] == "passed"
    assert payload["failures"] == {}
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_inventory_rejects_empty_or_duplicate_prefixes(tmp_path, monkeypatch) -> None:
    _credentials(monkeypatch)
    with pytest.raises(ValueError, match="non-empty"):
        s3_inventory.inventory(tmp_path / "one.json", [""])
    with pytest.raises(ValueError, match="unique"):
        s3_inventory.inventory(tmp_path / "two.json", ["same", "same"])
