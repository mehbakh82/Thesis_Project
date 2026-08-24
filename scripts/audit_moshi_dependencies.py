#!/usr/bin/env python3
"""Fail when the pinned Moshi dependency risks drift beyond reviewed exceptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "configs" / "moshi_dependency_risk_policy.json"
DEFAULT_LOCK = ROOT / "requirements-moshi.lock"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _expected_findings(policy: dict) -> tuple[dict[str, dict], list[str]]:
    errors: list[str] = []
    expected: dict[str, dict] = {}
    risks = policy.get("accepted_risks")
    if not isinstance(risks, list) or not risks:
        return {}, ["policy accepted_risks must be a non-empty list"]
    for risk in risks:
        if not isinstance(risk, dict):
            errors.append("every accepted risk must be an object")
            continue
        package = str(risk.get("package") or "").strip().lower()
        version = str(risk.get("version") or "").strip()
        advisories = risk.get("advisories")
        mitigations = risk.get("mitigations")
        if not package or package in expected:
            errors.append(f"invalid or duplicate policy package: {package!r}")
            continue
        if not version:
            errors.append(f"{package}: version is required")
        if not isinstance(advisories, list) or not advisories:
            errors.append(f"{package}: advisories must be a non-empty list")
            advisory_set: set[str] = set()
        else:
            advisory_set = {str(item).strip() for item in advisories if str(item).strip()}
            if len(advisory_set) != len(advisories):
                errors.append(f"{package}: advisories must be unique and non-empty")
        if not str(risk.get("upstream_constraint") or "").strip():
            errors.append(f"{package}: upstream_constraint is required")
        if not isinstance(mitigations, list) or not all(
            str(item).strip() for item in mitigations
        ):
            errors.append(f"{package}: non-empty mitigations are required")
        expected[package] = {"version": version, "advisories": advisory_set}
    return expected, errors


def _run_audit(lock_path: Path) -> tuple[dict, int, str]:
    command = [
        sys.executable,
        "-m",
        "pip_audit",
        "--requirement",
        str(lock_path),
        "--format",
        "json",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"pip-audit failed with exit {completed.returncode}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"pip-audit did not return valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("dependencies"), list):
        raise RuntimeError("pip-audit JSON is missing its dependency list")
    return payload, completed.returncode, completed.stderr.strip()


def _observed_findings(payload: dict) -> tuple[dict[str, dict], int]:
    observed: dict[str, dict] = {}
    raw_count = 0
    for dependency in payload["dependencies"]:
        vulnerabilities = dependency.get("vulns") or []
        if not vulnerabilities:
            continue
        package = str(dependency.get("name") or "").strip().lower()
        version = str(dependency.get("version") or "").strip()
        advisory_ids = {
            str(item.get("id") or "").strip()
            for item in vulnerabilities
            if str(item.get("id") or "").strip()
        }
        raw_count += len(vulnerabilities)
        observed[package] = {"version": version, "advisories": advisory_ids}
    return observed, raw_count


def _compare_findings(expected: dict[str, dict], observed: dict[str, dict]) -> list[str]:
    errors: list[str] = []
    for package in sorted(set(expected) | set(observed)):
        if package not in expected:
            errors.append(f"unreviewed vulnerable package: {package}")
            continue
        if package not in observed:
            errors.append(f"stale accepted risk no longer reported: {package}")
            continue
        if expected[package]["version"] != observed[package]["version"]:
            errors.append(
                f"{package}: version drift: expected {expected[package]['version']}, "
                f"observed {observed[package]['version']}"
            )
        added = observed[package]["advisories"] - expected[package]["advisories"]
        removed = expected[package]["advisories"] - observed[package]["advisories"]
        if added:
            errors.append(f"{package}: unreviewed advisories: {sorted(added)}")
        if removed:
            errors.append(f"{package}: stale advisories: {sorted(removed)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    policy = _load_object(args.policy)
    expected, errors = _expected_findings(policy)
    audit, audit_exit, audit_message = _run_audit(args.lock)
    observed, raw_count = _observed_findings(audit)

    errors.extend(_compare_findings(expected, observed))

    result = {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "scope": policy.get("scope"),
        "reviewed_at": policy.get("reviewed_at"),
        "policy_status": policy.get("status"),
        "policy_sha256": _sha256(args.policy),
        "lock_sha256": _sha256(args.lock),
        "audit_exit": audit_exit,
        "audit_message": audit_message,
        "vulnerable_packages": {
            package: {
                "version": finding["version"],
                "advisories": sorted(finding["advisories"]),
            }
            for package, finding in sorted(observed.items())
        },
        "unique_accepted_advisories": sum(
            len(finding["advisories"]) for finding in observed.values()
        ),
        "raw_audit_findings": raw_count,
        "errors": errors,
        "passed": not errors,
        "note": (
            "A passing status means every current finding exactly matches a reviewed, "
            "upstream-constrained exception; it does not mean the environment has no "
            "known vulnerabilities."
        ),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
