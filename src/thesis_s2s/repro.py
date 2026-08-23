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
SNAPSHOT_DIRS = ("configs", "docs", "scripts", "src", "tests")
SNAPSHOT_FILES = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "requirements-moshi.lock",
    "third_party/UPSTREAMS.lock.json",
    "results/corpus_audit.json",
    "results/prepared_episode_audit.json",
    "results/prepared_reserve_tabaghe16_audit.json",
    "results/conversation_selection_audit.json",
    "results/diarization_smoke_h100.json",
    "results/hardware/current_preflight.json",
    "results/hardware/moshi_environment.json",
    "results/conversation_rights_report.json",
    "results/conversation_source_authorization_report_combined.json",
    "results/conversation_noise_report.json",
    "results/conversation_reserve_tabaghe16_noise_report.json",
    "results/conversation_yield_estimate.json",
    "results/conversation_reserve_tabaghe16_yield_estimate.json",
    "results/conversation_reserve_yield_selection.json",
    "results/conversation_yield_estimate_combined.json",
    "results/diarized_episode_audit.json",
    "results/diarized_episode_audit_combined.json",
    "results/diarized_episode_audit_combined_authorized.json",
    "results/conversation_audit.json",
    "results/moshi_export_report.json",
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
            raise ValueError(
                f"upstream revision must be a full 40-character commit: {entry['name']}"
            )
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
    final_path = root / "results" / "conversation_audit.json"
    staging_path = root / "results" / "diarized_episode_audit_combined_authorized.json"
    yield_path = root / "results" / "conversation_yield_estimate_combined.json"
    status: dict[str, object] = {
        "present": final_path.is_file(),
        "thesis_coverage_ok": False,
        "path": str(final_path),
        "staging_audit_path": str(staging_path),
        "yield_estimate_path": str(yield_path),
    }
    if final_path.is_file():
        try:
            report = json.loads(final_path.read_text(encoding="utf-8"))
            status.update(
                {
                    "thesis_coverage_ok": report.get("thesis_coverage_ok") is True,
                    "pairs": report.get("pairs"),
                    "hours": report.get("hours"),
                }
            )
        except json.JSONDecodeError as exc:
            status["error"] = str(exc)

    if staging_path.is_file():
        try:
            staging = json.loads(staging_path.read_text(encoding="utf-8"))
            requirements = staging.get("requirements") or {}
            automatic = {
                key: value
                for key, value in requirements.items()
                if key != "manual_qa_sample_present"
            }
            status["staging"] = {
                "present": True,
                "windows": staging.get("windows"),
                "episodes": staging.get("episodes"),
                "automatic_multi_speaker_hours": staging.get("automatic_multi_speaker_hours"),
                "reference_aligned_hours": staging.get("reference_aligned_hours"),
                "automatic_gates_pass": bool(automatic) and all(automatic.values()),
                "manual_qa_complete": requirements.get("manual_qa_sample_present") is True,
            }
        except json.JSONDecodeError as exc:
            status["staging"] = {"present": True, "error": str(exc)}
    else:
        status["staging"] = {"present": False}

    if yield_path.is_file():
        try:
            estimate = json.loads(yield_path.read_text(encoding="utf-8"))
            status["yield_estimate"] = {
                "present": True,
                "pairs": estimate.get("estimated_pairs"),
                "hours": estimate.get("estimated_pair_hours"),
                "hours_100_to_200": estimate.get("hours_100_to_200") is True,
                "non_mutating": estimate.get("non_mutating_estimate") is True,
            }
        except json.JSONDecodeError as exc:
            status["yield_estimate"] = {"present": True, "error": str(exc)}
    else:
        status["yield_estimate"] = {"present": False}
    return status


def _nvidia_smi() -> dict:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return {
            "available": True,
            "fields": [
                "name",
                "driver_version",
                "memory_total_mib",
                "memory_used_mib",
                "memory_free_mib",
                "utilization_gpu_percent",
            ],
            "rows": output.splitlines(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)[:200]}


def _base_model_status(root: Path, entry: dict | None, training_config: dict) -> dict:
    model_paths = training_config.get("moshi_paths") or {}
    configured = {
        Path(value).name: root / value
        for key in ("moshi_path", "mimi_path", "tokenizer_path")
        if isinstance((value := model_paths.get(key)), str) and value
    }
    checks = {}
    for filename, expected in ((entry or {}).get("weights") or {}).items():
        path = configured.get(filename)
        actual_bytes = path.stat().st_size if path is not None and path.is_file() else None
        actual_sha256 = sha256_file(path) if path is not None and path.is_file() else None
        checks[filename] = {
            "path": str(path) if path is not None else None,
            "expected_bytes": expected.get("bytes"),
            "actual_bytes": actual_bytes,
            "expected_sha256": expected.get("sha256"),
            "actual_sha256": actual_sha256,
            "matches_pin": actual_bytes == expected.get("bytes")
            and actual_sha256 == expected.get("sha256"),
        }
    config_value = model_paths.get("config_path")
    config_path = root / config_value if isinstance(config_value, str) else None
    config_present = config_path is not None and config_path.is_file()
    return {
        "revision": (entry or {}).get("revision"),
        "files": checks,
        "architecture_config": str(config_path) if config_path is not None else None,
        "architecture_config_present": config_present,
        "valid": bool(checks)
        and all(row["matches_pin"] for row in checks.values())
        and config_present,
    }


def gpu_preflight(out_json: Path | None = None) -> dict:
    """Report independent training and target-hardware evaluation readiness."""

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
    upstream = verify_upstream_lock(checkouts_root=root / "third_party" / "checkouts")
    conversation = _conversation_status(root)
    training_config = load_yaml(root / "configs" / "moshi_h100.yaml")
    locked_upstreams = load_upstream_lock()["upstreams"]
    moshi_entry = next(
        (entry for entry in locked_upstreams if entry.get("name") == "Moshi-Finetune"),
        None,
    )
    voice_entry = next(
        (entry for entry in locked_upstreams if entry.get("name") == "Mana-Persian-Piper"),
        None,
    )
    base_entry = next(
        (entry for entry in locked_upstreams if entry.get("name") == "Moshika-PyTorch-BF16"),
        None,
    )
    base_model = _base_model_status(root, base_entry, training_config)
    voice_weights = (voice_entry or {}).get("weights") or {}
    voice_file = str(voice_weights.get("file") or "")
    voice_path = root / "models" / "piper" / voice_file if voice_file else None
    voice_expected_sha256 = str(voice_weights.get("sha256") or "")
    voice_actual_sha256 = (
        sha256_file(voice_path) if voice_path is not None and voice_path.is_file() else None
    )
    voice_pinned = bool(
        voice_actual_sha256
        and voice_expected_sha256
        and voice_actual_sha256 == voice_expected_sha256
    )
    trainer_path = root / "third_party" / "checkouts" / "moshi-finetune" / "train.py"
    official_trainer_declared = bool(
        moshi_entry and str(moshi_entry.get("training_entrypoint") or "").strip()
    )
    reviewed_trainer_available = official_trainer_declared and trainer_path.is_file()
    training_gates = {
        "cuda_available": torch_info.get("cuda_available") is True,
        "bf16_supported": torch_info.get("bf16_supported") is True,
        "upstream_revisions_pinned": upstream["valid"],
        "conversational_data_audit_passes": conversation["thesis_coverage_ok"],
        "reviewed_training_entrypoint_available": reviewed_trainer_available,
        "base_model_files_pinned": base_model["valid"],
        "assistant_voice_target_pinned": voice_pinned,
    }
    evaluation_gates = {
        "physical_gpu_12_to_24_gb": physical_eligible,
        "cuda_available": torch_info.get("cuda_available") is True,
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "nvidia_smi": _nvidia_smi(),
        "torch": torch_info,
        "is_rtx_4090": "4090" in device,
        "upstreams": upstream,
        "conversation_data": conversation,
        "training_config": str(root / "configs" / "moshi_h100.yaml"),
        "training_base_model": (training_config.get("moshi_paths") or {}).get("hf_repo_id"),
        "official_upstream_training_entrypoint_available": official_trainer_declared,
        "configured_trainer_entrypoint": str(trainer_path),
        "base_model": base_model,
        "assistant_voice_target": {
            "path": str(voice_path) if voice_path is not None else None,
            "expected_sha256": voice_expected_sha256 or None,
            "actual_sha256": voice_actual_sha256,
            "matches_pin": voice_pinned,
        },
        "training_gates": training_gates,
        "evaluation_gates": evaluation_gates,
        "training_hardware_ready": bool(
            training_gates["cuda_available"] and training_gates["bf16_supported"]
        ),
        "evaluation_hardware_ready": all(evaluation_gates.values()),
        "adaptation_run_ready": all(training_gates.values()),
        "adaptation_status": (
            "ready" if all(training_gates.values()) else "blocked_by_failed_training_gates"
        ),
        "note": (
            "The H100 is valid training hardware. A physical 12–24 GB GPU is required only "
            "for the final target-hardware fit and live-latency evidence. Moshi-Finetune is "
            "the pinned genuine response-audio trainer; data/QA gates still apply."
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
                path
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not any(part.endswith(".egg-info") for part in path.parts)
                and path.suffix not in {".pyc", ".orig", ".rej"}
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
