"""List 2TB (or 1TB) MinIO prefixes without embedding credentials."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path


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


def rclone_path(path: str) -> str:
    remote = configured_remote()
    if remote and path.startswith(":s3:"):
        return f"{remote}:{path.removeprefix(':s3:')}"
    return path


def rclone_env() -> dict[str, str]:
    remote = configured_remote()
    if remote:
        return {
            "endpoint": "configured-rclone-remote",
            "access": "",
            "secret": "",
            "bucket": os.environ.get("S3_BUCKET", "asr"),
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
        "endpoint": os.environ["S3_ENDPOINT"],
        "access": os.environ["S3_ACCESS_KEY_ID"],
        "secret": os.environ["S3_SECRET_ACCESS_KEY"],
        "bucket": os.environ.get("S3_BUCKET", "asr"),
        "provider": os.environ.get("S3_PROVIDER", "Minio"),
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
        cmd, check=True, capture_output=True, text=True, env=rclone_process_env()
    )
    return result.stdout


def ls(path: str, include: str | None = None) -> str:
    cmd = rclone_prefix() + ["ls", rclone_path(path)]
    if include:
        cmd.extend(["--include", include])
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=rclone_process_env()
    )
    return result.stdout


def inventory(out_json: str | Path, extra_prefixes: list[str] | None = None) -> dict:
    cfg = rclone_env()
    bucket = rclone_path(f":s3:{cfg['bucket']}")
    prefixes = extra_prefixes or []
    listing = lsd(bucket)
    extra: dict[str, object] = {}
    payload: dict[str, object] = {
        "bucket": cfg["bucket"],
        "endpoint": cfg["endpoint"],
        "top_level": listing.splitlines(),
        "note": (
            "All five YouTube sources are on the 2TB asr bucket. Tabaghe16 is under "
            "STT/YT_PodCast_Chunks/{Audio_Chunks,CSVs}/طبقه 16."
        ),
        "extra": extra,
    }
    for prefix in prefixes:
        try:
            extra[prefix] = lsd(prefix).splitlines()[:200]
        except subprocess.CalledProcessError as exc:
            extra[prefix] = {"error": exc.stderr}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload
