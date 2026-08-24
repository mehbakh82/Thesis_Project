from __future__ import annotations

from scripts.audit_moshi_dependencies import (
    DEFAULT_POLICY,
    _compare_findings,
    _expected_findings,
    _load_object,
    _observed_findings,
)


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

    assert observed["example"]["advisories"] == {"CVE-1"}
    assert raw_count == 2
