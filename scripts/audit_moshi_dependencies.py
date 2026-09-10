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
        allow_unreported = risk.get("allow_unreported_by_scanner", False)
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
        if not isinstance(allow_unreported, bool):
            errors.append(f"{package}: allow_unreported_by_scanner must be boolean")
        expected[package] = {
            "version": version,
            "advisories": advisory_set,
            "allow_unreported_by_scanner": allow_unreported,
        }
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
        raise RuntimeError(
            f"pip-audit failed with exit {completed.returncode}: {detail}"
        )
    if not completed.stdout.strip():
        detail = completed.stderr.strip() or "empty stdout"
        raise RuntimeError(f"pip-audit returned no JSON: {detail}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip()
        raise RuntimeError(
            f"pip-audit did not return valid JSON: {exc}; stderr: {detail}"
        ) from exc
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
        raw_count += len(vulnerabilities)
        finding = observed.setdefault(
            package,
            {"version": version, "advisories": set(), "advisory_identities": []},
        )
        if finding["version"] != version:
            # Preserve the ambiguity so the exact-version comparison fails
            # instead of silently accepting the last duplicate package row.
            finding["version"] = f"{finding['version']}|{version}"
        for vulnerability in vulnerabilities:
            primary = str(vulnerability.get("id") or "").strip()
            if not primary:
                continue
            identity = {primary}
            aliases = vulnerability.get("aliases") or []
            if isinstance(aliases, list):
                identity.update(str(item).strip() for item in aliases if str(item).strip())
            finding["advisories"].add(primary)

            # A scanner can emit duplicate database records for the same
            # vulnerability. Merge overlapping ID/alias groups so identity
            # comparison is stable if its preferred primary ID changes.
            overlapping = [
                group for group in finding["advisory_identities"] if group & identity
            ]
            if overlapping:
                merged = set(identity)
                for group in overlapping:
                    merged.update(group)
                    finding["advisory_identities"].remove(group)
                finding["advisory_identities"].append(merged)
            else:
                finding["advisory_identities"].append(identity)
    return observed, raw_count


def _resolved_versions(payload: dict) -> dict[str, str]:
    """Return every dependency version resolved by pip-audit, including clean rows."""

    return {
        str(dependency.get("name") or "").strip().lower(): str(
            dependency.get("version") or ""
        ).strip()
        for dependency in payload["dependencies"]
        if str(dependency.get("name") or "").strip()
    }


def _compare_findings(
    expected: dict[str, dict],
    observed: dict[str, dict],
    resolved_versions: dict[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    for package in sorted(set(expected) | set(observed)):
        if package not in expected:
            errors.append(f"unreviewed vulnerable package: {package}")
            continue
        if package not in observed:
            if not expected[package].get("allow_unreported_by_scanner"):
                errors.append(f"stale accepted risk no longer reported: {package}")
                continue
            if resolved_versions is None or package not in resolved_versions:
                errors.append(
                    f"{package}: scanner-unmapped risk lacks resolved package evidence"
                )
                continue
            if expected[package]["version"] != resolved_versions[package]:
                errors.append(
                    f"{package}: version drift: expected {expected[package]['version']}, "
                    f"resolved {resolved_versions[package]}"
                )
            continue
        if expected[package]["version"] != observed[package]["version"]:
            errors.append(
                f"{package}: version drift: expected {expected[package]['version']}, "
                f"observed {observed[package]['version']}"
            )
        expected_ids = expected[package]["advisories"]
        identities = observed[package].get("advisory_identities")
        if not isinstance(identities, list):
            identities = [{item} for item in observed[package]["advisories"]]
        matched_expected: set[str] = set()
        added: set[str] = set()
        for identity in identities:
            identity_set = {str(item) for item in identity}
            matches = identity_set & expected_ids
            if matches:
                matched_expected.update(matches)
            else:
                primary_matches = identity_set & observed[package]["advisories"]
                added.add(sorted(primary_matches or identity_set)[0])
        removed = expected_ids - matched_expected
        if added:
            errors.append(f"{package}: unreviewed advisories: {sorted(added)}")
        if removed:
            errors.append(f"{package}: stale advisories: {sorted(removed)}")
    return errors


def main(
    *, default_policy: Path = DEFAULT_POLICY, default_lock: Path = DEFAULT_LOCK
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, default=default_policy)
    parser.add_argument("--lock", type=Path, default=default_lock)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    policy = _load_object(args.policy)
    expected, errors = _expected_findings(policy)
    audit, audit_exit, audit_message = _run_audit(args.lock)
    observed, raw_count = _observed_findings(audit)
    resolved_versions = _resolved_versions(audit)

    errors.extend(_compare_findings(expected, observed, resolved_versions))
    known_unmapped = {
        package: finding
        for package, finding in expected.items()
        if finding.get("allow_unreported_by_scanner") and package not in observed
    }

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
                "advisory_identities": [
                    sorted(identity) for identity in finding["advisory_identities"]
                ],
            }
            for package, finding in sorted(observed.items())
        },
        "known_scanner_unmapped_risks": {
            package: {
                "resolved_version": resolved_versions.get(package),
                "advisories": sorted(finding["advisories"]),
            }
            for package, finding in sorted(known_unmapped.items())
        },
        "reviewed_package_versions": {
            package: resolved_versions.get(package) for package in sorted(expected)
        },
        "unique_accepted_advisories": sum(
            len(finding["advisories"]) for finding in expected.values()
        ),
        "raw_audit_findings": raw_count,
        "errors": errors,
        "passed": not errors,
        "note": (
            "A passing status means every scanner-reported finding and every explicitly "
            "known scanner-unmapped risk matches a reviewed exact-version exception; it "
            "does not mean the environment has no known vulnerabilities."
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
