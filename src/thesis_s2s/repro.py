"""Reproducibility, upstream-lock, and physical-GPU readiness reports."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from thesis_s2s.config import load_yaml, project_root
from thesis_s2s.metrics import gpu_inventory, write_json

SHA256_RE = re.compile(r"^[0-9a-f]{40}$")
SNAPSHOT_DIRS = ("configs", "docs", "src", "tests")
SNAPSHOT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "third_party/UPSTREAMS.lock.json",
    "results/corpus_audit.json",
    "results/conversation_audit.json",
    "results/bargein/heldout_report.json",
    "results/bargein/recorded_heldout_report.json",
    "results/bargein/feature_heldout_report.json",
    "results/eval/human_study.json",
)
SNAPSHOT_GLOBS = (
    "models/piper/*",
    "checkpoints/llama_omni2_fa/*.pt",
    "results/bargein/*.pkl",
)
PACKAGE_NAMES = (
    "accelerate",
    "fastapi",
    "numpy",
    "peft",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "uvicorn",
    "soundfile",
    "piper-tts",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream_lock(path: Path | None = None) -> dict:
    lock_path = Path(path or project_root() / "third_party" / "UPSTREAMS.lock.json")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("upstreams"), list):
        raise ValueError("unsupported upstream lock schema")
    names: set[str] = set()
    for entry in payload["upstreams"]:
        required = ("name", "repository", "revision", "license")
        if not all(isinstance(entry.get(key), str) and entry[key] for key in required):
            raise ValueError(f"upstream entry is incomplete: {entry!r}")
        if not SHA256_RE.fullmatch(entry["revision"]):
            raise ValueError(f"upstream revision must be a full 40-character commit: {entry['name']}")
        if entry["name"] in names:
            raise ValueError(f"duplicate upstream name: {entry['name']}")
        names.add(entry["name"])
    return payload


def verify_upstream_lock(
    path: Path | None = None,
    *,
    checkouts_root: Path | None = None,
) -> dict:
    """Validate immutable pins and optionally compare local checkout HEADs."""

    payload = load_upstream_lock(path)
    checks = []
    for entry in payload["upstreams"]:
        check = {
            "name": entry["name"],
            "revision": entry["revision"],
            "pin_valid": True,
            "checkout_present": None,
            "checkout_matches": None,
        }
        if checkouts_root is not None:
            checkout = Path(checkouts_root) / entry["checkout_dir"]
            check["checkout_present"] = checkout.is_dir()
            if checkout.is_dir():
                try:
                    head = subprocess.run(
                        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    ).stdout.strip()
                    check["checkout_head"] = head
                    check["checkout_matches"] = head == entry["revision"]
                except (OSError, subprocess.SubprocessError) as exc:
                    check["checkout_error"] = str(exc)[:200]
                    check["checkout_matches"] = False
        checks.append(check)
    return {
        "schema_version": 1,
        "valid": all(row["pin_valid"] and row["checkout_matches"] is not False for row in checks),
        "checks": checks,
    }


def _conversation_status(root: Path) -> dict:
    path = root / "results" / "conversation_audit.json"
    if not path.is_file():
        return {"present": False, "thesis_coverage_ok": False, "path": str(path)}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"present": True, "thesis_coverage_ok": False, "error": str(exc)}
    return {
        "present": True,
        "thesis_coverage_ok": report.get("thesis_coverage_ok") is True,
        "pairs": report.get("pairs"),
        "hours": report.get("hours"),
        "path": str(path),
    }


def _nvidia_smi() -> dict:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {"available": True, "rows": output.splitlines()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)[:200]}


def gpu_preflight(out_json: Path | None = None) -> dict:
    """Describe whether this is a physical thesis-eligible 12–24 GB run."""

    root = project_root()
    gpu = gpu_inventory()
    torch_info: dict[str, object] = {}
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        }
    except Exception as exc:  # pragma: no cover - depends on deployment
        torch_info = {"error": str(exc)[:200]}
    device = str(gpu.get("device") or "")
    physical_eligible = bool(gpu.get("official_size"))
    upstream = verify_upstream_lock()
    conversation = _conversation_status(root)
    adaptation_config = load_yaml(root / "configs" / "llama_omni2_4090.yaml")
    trainer_value = (adaptation_config.get("adaptation") or {}).get("trainer_entrypoint")
    trainer_path = root / str(trainer_value) if trainer_value else None
    reviewed_trainer_available = trainer_path is not None and trainer_path.is_file()
    gates = {
        "physical_gpu_12_to_24_gb": physical_eligible,
        "cuda_available": torch_info.get("cuda_available") is True,
        "bf16_supported": torch_info.get("bf16_supported") is True,
        "upstream_revisions_pinned": upstream["valid"],
        "conversational_data_audit_passes": conversation["thesis_coverage_ok"],
        "reviewed_training_entrypoint_available": reviewed_trainer_available,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "nvidia_smi": _nvidia_smi(),
        "torch": torch_info,
        "is_rtx_4090": "4090" in device,
        "upstreams": upstream,
        "conversation_data": conversation,
        "official_upstream_training_entrypoint_available": False,
        "configured_trainer_entrypoint": str(trainer_path) if trainer_path is not None else None,
        "gates": gates,
        "evaluation_hardware_ready": physical_eligible,
        "adaptation_run_ready": all(gates.values()),
        "adaptation_status": "ready" if all(gates.values()) else "blocked_by_failed_gates",
        "note": (
            "A 4090 is eligible physical evaluation hardware. The pinned LLaMA-Omni2 release "
            "does not provide a complete language-model training entrypoint, so this report "
            "must not be interpreted as a completed or reproducible Persian adaptation."
        ),
    }
    if out_json is not None:
        write_json(out_json, report)
    return report


def _git_state(root: Path) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"available": True, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)[:200]}


def release_snapshot(out_json: Path | None = None) -> dict:
    """Hash reproducibility-critical code, docs, configs, and evidence reports."""

    root = project_root()
    paths: set[Path] = set()
    for directory in SNAPSHOT_DIRS:
        base = root / directory
        if base.is_dir():
            paths.update(
                path for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix not in {".pyc", ".orig", ".rej"}
            )
    paths.update(root / name for name in SNAPSHOT_FILES if (root / name).is_file())
    for pattern in SNAPSHOT_GLOBS:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    files = {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(paths)
    }
    packages: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git": _git_state(root),
        "gpu": gpu_inventory(),
        "nvidia_smi": _nvidia_smi(),
        "packages": packages,
        "files": files,
        "file_count": len(files),
    }
    target = Path(out_json or root / "results" / "release" / "snapshot.json")
    write_json(target, report)
    return report
