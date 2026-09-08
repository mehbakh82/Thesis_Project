from pathlib import Path

import yaml

from thesis_s2s.eval.model_selection import audit_model_selection


def _evidence(path: Path) -> None:
    path.write_text("evidence\n", encoding="utf-8")


def test_model_selection_requires_controlled_multiple_eligible_candidates(tmp_path):
    _evidence(tmp_path / "evidence.txt")
    catalog = {
        "as_of": "2026-09-08",
        "tracks": {
            "responder": {
                "current_selection": "a",
                "controlled_same_panel_comparison": True,
                "requirements": {
                    "documented_capability": {"minimum": "documented"},
                    "project_quality": {"minimum": "verified"},
                },
                "candidates": {
                    name: {
                        "evidence": {
                            "documented_capability": {
                                "status": "documented",
                                "source": "https://example.test/model",
                            },
                            "project_quality": {
                                "status": "verified",
                                "source": "local:evidence.txt",
                            },
                        }
                    }
                    for name in ("a", "b")
                },
            }
        },
    }
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog), encoding="utf-8")

    report = audit_model_selection(
        catalog_path,
        tmp_path / "report.json",
        root=tmp_path,
    )

    track = report["tracks"]["responder"]
    assert track["eligible_candidates"] == ["a", "b"]
    assert track["catalog_comparison_complete"] is True
    assert report["model_selection_audit"]["passes"] is True
    assert report["model_selection_audit"]["catalog_bounded_selection_claim_allowed"] is True
    assert report["model_selection_audit"]["global_best_model_claim_allowed"] is False


def test_unknown_or_missing_local_evidence_fails_closed(tmp_path):
    catalog = {
        "tracks": {
            "asr": {
                "current_selection": "candidate",
                "controlled_same_panel_comparison": False,
                "requirements": {"quality": {"minimum": "verified"}},
                "candidates": {
                    "candidate": {
                        "evidence": {
                            "quality": {
                                "status": "verified",
                                "source": "local:missing.json",
                            }
                        }
                    }
                },
            }
        }
    }
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog), encoding="utf-8")

    report = audit_model_selection(
        catalog_path,
        tmp_path / "report.json",
        root=tmp_path,
    )

    candidate = report["tracks"]["asr"]["candidates"]["candidate"]
    assert candidate["eligible"] is False
    assert candidate["criteria"]["quality"]["source"]["present"] is False
    assert report["model_selection_audit"]["passes"] is False
