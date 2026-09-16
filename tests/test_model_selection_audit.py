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


def test_project_catalog_eliminates_post_release_candidates_before_download(tmp_path):
    root = Path(__file__).resolve().parents[1]
    report = audit_model_selection(
        root / "configs/model_selection_audit.yaml",
        tmp_path / "report.json",
        root=root,
    )

    direct = report["tracks"]["direct_s2s"]
    lychee = direct["candidates"]["lychee-fd"]
    assert lychee["eligible"] is False
    assert lychee["criteria"]["native_full_duplex"]["meets"] is True
    assert lychee["criteria"]["public_inference"]["meets"] is True
    assert lychee["criteria"]["public_adaptation"]["meets"] is True
    assert lychee["criteria"]["persian_speech_output"]["status"] == "unknown"
    assert lychee["criteria"]["persian_speech_output"]["meets"] is False
    assert lychee["criteria"]["target_24gb_path"]["status"] == "failed"
    assert lychee["criteria"]["target_24gb_path"]["meets"] is False
    assert "lychee-fd" not in direct["eligible_candidates"]

    for candidate in direct["candidates"].values():
        assert "public_license_or_terms" in candidate["criteria"]

    nemotron = direct["candidates"]["nvidia-nemotronlabs-voicechat-11b"]
    assert nemotron["criteria"]["persian_speech_output"]["status"] == "failed"
    assert nemotron["criteria"]["target_24gb_path"]["status"] == "failed"
    assert nemotron["eligible"] is False

    venus = direct["candidates"]["realtime-venus-audio-9b"]
    assert venus["criteria"]["native_full_duplex"]["meets"] is True
    assert venus["criteria"]["public_inference"]["status"] == "unknown"
    assert venus["eligible"] is False

    duplexsla = direct["candidates"]["duplexsla"]
    assert duplexsla["criteria"]["public_inference"]["status"] == "failed"
    assert duplexsla["eligible"] is False
