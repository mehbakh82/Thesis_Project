from __future__ import annotations

from pathlib import Path

from scripts.audit_moshi_dependencies import (
    DEFAULT_POLICY,
    _compare_findings,
    _expected_findings,
    _load_object,
    _observed_findings,
    _resolved_versions,
)

APPLICATION_POLICY = Path("configs/application_dependency_risk_policy.json")


def test_reviewed_moshi_dependency_policy_is_well_formed() -> None:
    expected, errors = _expected_findings(_load_object(DEFAULT_POLICY))

    assert errors == []
    assert set(expected) == {"aiohttp", "sentencepiece", "torch"}
    assert all(finding["advisories"] for finding in expected.values())


def test_dependency_risk_comparison_fails_closed_on_drift() -> None:
    expected, _ = _expected_findings(_load_object(DEFAULT_POLICY))
    observed = {
        package: {
            "version": finding["version"],
            "advisories": set(finding["advisories"]),
        }
        for package, finding in expected.items()
    }
    assert _compare_findings(expected, observed) == []

    observed["torch"]["advisories"].add("PYSEC-FUTURE-NEW")
    errors = _compare_findings(expected, observed)
    assert any("unreviewed advisories" in error for error in errors)


def test_reviewed_application_dependency_policy_is_narrow() -> None:
    expected, errors = _expected_findings(_load_object(APPLICATION_POLICY))

    assert errors == []
    assert expected == {
        "accelerate": {
            "version": "1.15.0",
            "advisories": {"CVE-2026-69112"},
            "allow_unreported_by_scanner": True,
        }
    }


def test_known_scanner_unmapped_risk_requires_exact_resolved_version() -> None:
    expected, errors = _expected_findings(_load_object(APPLICATION_POLICY))
    assert errors == []

    assert _compare_findings(expected, {}, {"accelerate": "1.15.0"}) == []
    drift = _compare_findings(expected, {}, {"accelerate": "1.16.0"})
    assert any("version drift" in error for error in drift)
    missing = _compare_findings(expected, {}, {})
    assert any("lacks resolved package evidence" in error for error in missing)


def test_observed_findings_deduplicates_rows_but_counts_raw_rows() -> None:
    payload = {
        "dependencies": [
            {
                "name": "example",
                "version": "1.0",
                "vulns": [{"id": "CVE-1"}, {"id": "CVE-1"}],
            }
        ]
    }

    observed, raw_count = _observed_findings(payload)
    resolved = _resolved_versions(payload)

    assert observed["example"]["advisories"] == {"CVE-1"}
    assert raw_count == 2
    assert resolved == {"example": "1.0"}
