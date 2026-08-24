from __future__ import annotations

import hashlib
import json
from pathlib import Path

from thesis_s2s.repro import _verify_tree_manifest

ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_moshi_client_security_overlay_and_attestation_are_bound() -> None:
    config = _json(ROOT / "configs" / "moshi_client_build.json")
    package_path = ROOT / config["overlay_package"]
    lock_path = ROOT / config["overlay_package_lock"]
    package = _json(package_path)
    lock = _json(lock_path)
    assert config["policy"]["build_time_critical_advisories_must_be_zero"] is True
    report = _json(ROOT / config["report"])

    assert config["node_image"].count("@sha256:") == 1
    assert _sha256(package_path) == config["overlay_package_sha256"]
    assert _sha256(lock_path) == config["overlay_package_lock_sha256"]
    assert package["dependencies"]["react-router-dom"] == "7.18.2"
    assert package["dependencies"]["ws"] == "8.21.3"
    assert lock["packages"]["node_modules/react-router-dom"]["version"] == "7.18.2"
    assert lock["packages"]["node_modules/ws"]["version"] == "8.21.3"
    assert lock["packages"]["node_modules/vite"]["version"] == "5.4.21"
    assert lock["packages"]["node_modules/vite-plugin-top-level-await"]["version"] == "1.4.1"
    assert lock["packages"]["node_modules/@swc/core"]["version"] == "1.5.7"

    assert report["valid"] is True
    assert report["security"]["production_audit_clean"] is True
    assert report["security"]["production_audit"]["vulnerabilities"]["total"] == 0
    assert report["source"]["overlay_package_sha256"] == _sha256(package_path)
    assert report["source"]["overlay_package_lock_sha256"] == _sha256(lock_path)
    assert report["versions"]["react_router_dom"] == "7.18.2"
    assert report["versions"]["ws"] == "8.21.3"
    assert len(report["dist"]["files"]) > 0
    assert len(report["dist"]["tree_sha256"]) == 64


def test_tree_attestation_rejects_an_undeclared_file(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("index", encoding="utf-8")
    (assets / "app.js").write_text("app", encoding="utf-8")
    rows = []
    tree_digest = hashlib.sha256()
    for path in sorted(item for item in dist.rglob("*") if item.is_file()):
        relative = path.relative_to(dist).as_posix()
        file_hash = _sha256(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_hash})
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(file_hash.encode("ascii"))
        tree_digest.update(b"\n")

    valid, actual = _verify_tree_manifest(dist, rows, tree_digest.hexdigest())
    assert valid is True
    assert actual == tree_digest.hexdigest()

    (assets / "injected.js").write_text("unexpected", encoding="utf-8")
    valid, _ = _verify_tree_manifest(dist, rows, tree_digest.hexdigest())
    assert valid is False
