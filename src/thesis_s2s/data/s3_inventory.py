"""List 2TB (or 1TB) MinIO prefixes without embedding credentials."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_RCLONE_TIMEOUT_SECONDS = 3600.0


def _load_alias_into_env(alias: str = "s3-2t") -> None:
    """Aliases are intentionally unsupported because their commands expose keys."""
    raise RuntimeError(
        f"shell alias credential extraction is disabled ({alias}); set S3_* variables"
    )


def configured_remote() -> str | None:
    value = os.environ.get("THESIS_RCLONE_REMOTE", "").strip()
    if not value:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("THESIS_RCLONE_REMOTE must be a simple configured remote name")
    return value


def rclone_timeout_seconds() -> float:
    raw = os.environ.get("RCLONE_TIMEOUT_SECONDS", str(DEFAULT_RCLONE_TIMEOUT_SECONDS))
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("RCLONE_TIMEOUT_SECONDS must be a positive finite number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("RCLONE_TIMEOUT_SECONDS must be a positive finite number")
    return timeout


def _validated_endpoint(value: str) -> str:
    endpoint = value.strip()
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "S3_ENDPOINT must be an http(s) URL without credentials, query, or fragment"
        )
    return endpoint


def _validated_bucket(value: str) -> str:
    bucket = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,62}", bucket):
        raise ValueError("S3_BUCKET must be a simple bucket name")
    return bucket


def _validated_provider(value: str) -> str:
    provider = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", provider):
        raise ValueError("S3_PROVIDER must be a simple provider name")
    return provider


def rclone_path(path: str) -> str:
    remote = configured_remote()
    if remote and path.startswith(":s3:"):
        return f"{remote}:{path.removeprefix(':s3:')}"
    return path


def rclone_env() -> dict[str, str]:
    remote = configured_remote()
    bucket = _validated_bucket(os.environ.get("S3_BUCKET", "asr"))
    if remote:
        return {
            "endpoint": "configured-rclone-remote",
            "access": "",
            "secret": "",
            "bucket": bucket,
            "provider": "configured",
            "remote": remote,
        }
    required = ["S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_ENDPOINT"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "S3 credentials must come from environment variables "
            f"(missing {missing}). Use your administrator-provided S3/rclone setup; "
            "do not copy keys into this repository."
        )
    return {
        "endpoint": _validated_endpoint(os.environ["S3_ENDPOINT"]),
        "access": os.environ["S3_ACCESS_KEY_ID"],
        "secret": os.environ["S3_SECRET_ACCESS_KEY"],
        "bucket": bucket,
        "provider": _validated_provider(os.environ.get("S3_PROVIDER", "Minio")),
    }


def rclone_prefix() -> list[str]:
    # A named remote uses rclone's existing protected config. Otherwise validate
    # environment credentials, but never put them in argv (visible via ps/procfs).
    if configured_remote():
        return ["rclone"]
    rclone_env()
    return ["rclone", "--config", "/dev/null"]


def rclone_process_env() -> dict[str, str]:
    cfg = rclone_env()
    env = os.environ.copy()
    if configured_remote():
        return env
    # On-the-fly backends such as :s3:bucket/path read RCLONE_S3_*.
    # Keep credentials in the child environment so they never appear in argv.
    env.update(
        {
            "RCLONE_S3_PROVIDER": cfg["provider"],
            "RCLONE_S3_ACCESS_KEY_ID": cfg["access"],
            "RCLONE_S3_SECRET_ACCESS_KEY": cfg["secret"],
            "RCLONE_S3_ENDPOINT": cfg["endpoint"],
            "RCLONE_S3_FORCE_PATH_STYLE": "true",
        }
    )
    return env


def lsd(path: str) -> str:
    cmd = rclone_prefix() + ["lsd", rclone_path(path)]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=rclone_process_env(),
        timeout=rclone_timeout_seconds(),
    )
    return result.stdout


def ls(path: str, include: str | None = None) -> str:
    cmd = rclone_prefix() + ["ls", rclone_path(path)]
    if include:
        cmd.extend(["--include", include])
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=rclone_process_env(),
        timeout=rclone_timeout_seconds(),
    )
    return result.stdout


def _failure_record(exc: BaseException) -> dict[str, object]:
    record: dict[str, object] = {
        "status": "failed",
        "error_type": type(exc).__name__,
    }
    if isinstance(exc, subprocess.CalledProcessError):
        record["returncode"] = int(exc.returncode)
    if isinstance(exc, subprocess.TimeoutExpired):
        record["timeout_seconds"] = float(exc.timeout)
    return record


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def inventory(out_json: str | Path, extra_prefixes: list[str] | None = None) -> dict:
    cfg = rclone_env()
    bucket = rclone_path(f":s3:{cfg['bucket']}")
    prefixes = extra_prefixes or []
    if any(not isinstance(prefix, str) or not prefix.strip() for prefix in prefixes):
        raise ValueError("extra inventory prefixes must be non-empty strings")
    if len(set(prefixes)) != len(prefixes):
        raise ValueError("extra inventory prefixes must be unique")
    extra: dict[str, object] = {}
    failures: dict[str, object] = {}
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "bucket": cfg["bucket"],
        "endpoint": cfg["endpoint"],
        "top_level": [],
        "note": (
            "All five YouTube sources are on the 2TB asr bucket. Tabaghe16 is under "
            "STT/YT_PodCast_Chunks/{Audio_Chunks,CSVs}/طبقه 16."
        ),
        "extra": extra,
        "failures": failures,
    }
    try:
        payload["top_level"] = lsd(bucket).splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        failures["top_level"] = _failure_record(exc)
    for prefix in prefixes:
        try:
            extra[prefix] = lsd(prefix).splitlines()[:200]
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            failure = _failure_record(exc)
            extra[prefix] = failure
            failures[f"extra:{prefix}"] = failure
    payload["status"] = "failed" if failures else "passed"
    output_path = Path(out_json)
    _write_json_atomic(output_path, payload)
    if failures:
        raise RuntimeError(f"S3 inventory incomplete; inspect the redacted report at {output_path}")
    return payload
