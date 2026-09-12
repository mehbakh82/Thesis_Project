#!/usr/bin/env python3
"""Build and attest the pinned Moshi web client with a security overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "moshi_client_build.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def require_hash(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"hash mismatch for {path}: expected {expected}, found {actual}")


def command_timeout_seconds() -> float:
    raw = os.environ.get("MOSHI_CLIENT_TIMEOUT_SECONDS", "1800")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError("MOSHI_CLIENT_TIMEOUT_SECONDS must be positive and finite") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("MOSHI_CLIENT_TIMEOUT_SECONDS must be positive and finite")
    return timeout


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=command_timeout_seconds(),
    )
    if check and result.returncode != 0:
        detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n{detail}")
    return result


def safe_build_dir(relative: str) -> Path:
    candidate = (ROOT / relative).resolve()
    allowed = (ROOT / "data" / "processed").resolve()
    if candidate == allowed or allowed not in candidate.parents:
        raise RuntimeError("build_dir must be a child of data/processed")
    if candidate.is_symlink():
        raise RuntimeError("refusing a symlinked build directory")
    return candidate


def tree_manifest(root: Path) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        rows.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return rows, digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    lock = load_json(ROOT / "third_party" / "UPSTREAMS.lock.json")
    upstream = next(
        (row for row in lock.get("upstreams", []) if row.get("name") == config["upstream_name"]),
        None,
    )
    if upstream is None:
        raise RuntimeError(f"upstream is not locked: {config['upstream_name']}")

    checkout = ROOT / "third_party" / "checkouts" / "moshi"
    client = checkout / str(config["client_subdir"])
    revision = run(["git", "-C", str(checkout), "rev-parse", "HEAD"]).stdout.strip()
    if revision != upstream.get("revision"):
        raise RuntimeError(
            f"Moshi checkout revision mismatch: {revision} != {upstream.get('revision')}"
        )

    source_package = client / "package.json"
    source_lock = client / "package-lock.json"
    overlay_package = ROOT / str(config["overlay_package"])
    overlay_lock = ROOT / str(config["overlay_package_lock"])
    require_hash(source_package, str(config["upstream_package_sha256"]))
    require_hash(source_lock, str(config["upstream_package_lock_sha256"]))
    require_hash(overlay_package, str(config["overlay_package_sha256"]))
    require_hash(overlay_lock, str(config["overlay_package_lock_sha256"]))

    verified = {
        "upstream_revision": revision,
        "upstream_package_sha256": sha256_file(source_package),
        "upstream_package_lock_sha256": sha256_file(source_lock),
        "overlay_package_sha256": sha256_file(overlay_package),
        "overlay_package_lock_sha256": sha256_file(overlay_lock),
    }
    if args.verify_only:
        print(json.dumps({"verified": True, **verified}, indent=2))
        return 0

    build_dir = safe_build_dir(str(config["build_dir"]))
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(
        client,
        build_dir,
        ignore=shutil.ignore_patterns("node_modules", "dist", ".git"),
    )
    shutil.copy2(overlay_package, build_dir / "package.json")
    shutil.copy2(overlay_lock, build_dir / "package-lock.json")

    image = str(config["node_image"])
    user = f"{os.getuid()}:{os.getgid()}"
    docker = [
        "docker",
        "run",
        "--rm",
        "--user",
        user,
        "--volume",
        f"{build_dir}:/app",
        "--workdir",
        "/app",
        image,
    ]
    install = run([*docker, "npm", "ci"])
    production_audit_result = run(
        [*docker, "npm", "audit", "--omit=dev", "--json"],
        check=False,
    )
    production_audit = json.loads(production_audit_result.stdout)
    production_total = production_audit.get("metadata", {}).get("vulnerabilities", {}).get("total")
    if production_audit_result.returncode != 0 or production_total != 0:
        raise RuntimeError(
            "Moshi client production dependency audit is not clean: "
            f"returncode={production_audit_result.returncode}, total={production_total}"
        )

    full_audit_result = run([*docker, "npm", "audit", "--json"], check=False)
    full_audit = json.loads(full_audit_result.stdout)
    full_vulnerabilities = full_audit.get("metadata", {}).get("vulnerabilities", {})
    if (
        config.get("policy", {}).get("build_time_critical_advisories_must_be_zero") is True
        and full_vulnerabilities.get("critical") != 0
    ):
        raise RuntimeError("Moshi client build graph has a critical advisory")
    build = run([*docker, "npm", "run", "build"])
    dist = build_dir / "dist"
    if not (dist / "index.html").is_file():
        raise RuntimeError("Moshi client build did not produce dist/index.html")
    dist_files, dist_tree_hash = tree_manifest(dist)

    image_identity = run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"]
    ).stdout.strip()
    overlay_payload = load_json(overlay_package)
    overlay_lock_payload = load_json(overlay_lock)
    packages = overlay_lock_payload.get("packages", {})
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "scientific_evidence": False,
        "source": {
            **verified,
            "client_subdir": str(config["client_subdir"]),
        },
        "build_environment": {
            "node_image": image,
            "image_id": image_identity,
            "uid_gid": user,
            "npm_ci_completed": install.returncode == 0,
        },
        "security": {
            "production_audit_clean": production_total == 0,
            "production_audit": production_audit.get("metadata", {}),
            "full_graph_audit": full_audit.get("metadata", {}),
            "full_graph_vulnerabilities": full_audit.get("vulnerabilities", {}),
            "policy": config.get("policy", {}),
            "note": (
                "Only the production graph is served. Remaining full-graph findings "
                "are build-time dependencies used with pinned trusted inputs in an "
                "ephemeral container; npm's development server is prohibited."
            ),
        },
        "versions": {
            "react_router_dom": overlay_payload.get("dependencies", {}).get("react-router-dom"),
            "ws": overlay_payload.get("dependencies", {}).get("ws"),
            "vite": packages.get("node_modules/vite", {}).get("version"),
            "top_level_await": packages.get("node_modules/vite-plugin-top-level-await", {}).get(
                "version"
            ),
            "swc_core": packages.get("node_modules/@swc/core", {}).get("version"),
        },
        "dist": {
            "path": str(dist.relative_to(ROOT)),
            "files": dist_files,
            "tree_sha256": dist_tree_hash,
        },
        "build_output_tail": build.stdout.strip().splitlines()[-30:],
        "build_warning_tail": build.stderr.strip().splitlines()[-30:],
    }
    report_path = ROOT / str(config["report"])
    write_json_atomic(report_path, report)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
