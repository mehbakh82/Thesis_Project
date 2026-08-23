"""List 2TB (or 1TB) MinIO prefixes without embedding credentials."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _load_alias_into_env(alias: str = "s3-2t") -> None:
    """Aliases are intentionally unsupported because their commands expose keys."""
    raise RuntimeError(
        f"shell alias credential extraction is disabled ({alias}); set S3_* variables"
    )


def rclone_env() -> dict[str, str]:
    required = ["S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_ENDPOINT"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "S3 credentials must come from environment variables "
            f"(missing {missing}). See /mnt/md0/utils/s3_utils/s3-cheatsheet.md; "
            "do not copy keys into this repo."
        )
    return {
        "endpoint": os.environ["S3_ENDPOINT"],
        "access": os.environ["S3_ACCESS_KEY_ID"],
        "secret": os.environ["S3_SECRET_ACCESS_KEY"],
        "bucket": os.environ.get("S3_BUCKET", "asr"),
        "provider": os.environ.get("S3_PROVIDER", "Minio"),
    }


def rclone_prefix() -> list[str]:
    # Validate credentials, but never put them in argv (visible via ps/procfs).
    rclone_env()
    return ["rclone", "--config", "/dev/null"]


def rclone_process_env() -> dict[str, str]:
    cfg = rclone_env()
    env = os.environ.copy()
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
    cmd = rclone_prefix() + ["lsd", path]
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=rclone_process_env()
    )
    return result.stdout


def ls(path: str, include: str | None = None) -> str:
    cmd = rclone_prefix() + ["ls", path]
    if include:
        cmd.extend(["--include", include])
    result = subprocess.run(
        cmd, check=True, capture_output=True, text=True, env=rclone_process_env()
    )
    return result.stdout


def inventory(out_json: str | Path, extra_prefixes: list[str] | None = None) -> dict:
    cfg = rclone_env()
    bucket = f":s3:{cfg['bucket']}"
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
