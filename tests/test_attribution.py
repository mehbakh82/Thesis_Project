from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_citation_metadata_identifies_repository_and_author() -> None:
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))

    assert citation["cff-version"] == "1.2.0"
    assert citation["repository-code"] == "https://github.com/mehbakh82/Thesis_Project"
    assert citation["authors"] == [
        {"given-names": "Mehran", "family-names": "Bakhtiari"}
    ]


def test_every_locked_upstream_is_revisioned_in_third_party_notices() -> None:
    lock = json.loads(
        (ROOT / "third_party/UPSTREAMS.lock.json").read_text(encoding="utf-8")
    )
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    assert lock["verified_at"] == "2026-09-06"
    assert len(lock["upstreams"]) == 10
    for upstream in lock["upstreams"]:
        assert upstream["name"] in notices
        assert upstream["revision"] in notices

    assert "`piper-tts==1.7.0`" in notices
    assert "GPL-3.0-or-later" in notices
    assert "fine-tuned Persian ASR artifact" in notices


def test_qwen_lock_matches_the_exact_positive_validation_runtime() -> None:
    lock = json.loads(
        (ROOT / "third_party/UPSTREAMS.lock.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (
            ROOT / "results/eval/cascade_real_service_validation_panel.json"
        ).read_text(encoding="utf-8")
    )
    qwen = next(
        row for row in lock["upstreams"] if row["name"] == "Qwen2.5-0.5B-Instruct"
    )
    responder = validation["components"]["responder"]

    assert qwen["repository"].endswith(responder["requested_model"])
    assert qwen["revision"] == responder["revision"]
    assert qwen["license"] == "Apache-2.0"


def test_qwen4b_lock_matches_the_frozen_final_test() -> None:
    lock = json.loads(
        (ROOT / "third_party/UPSTREAMS.lock.json").read_text(encoding="utf-8")
    )
    result = json.loads(
        (ROOT / "results/eval/qwen4b_responder_v2_final_test_proxy.json").read_text(
            encoding="utf-8"
        )
    )
    qwen = next(
        row for row in lock["upstreams"] if row["name"] == "Qwen3-4B-Instruct-2507"
    )
    responder = result["responders"]["qwen4b_v2"]

    assert qwen["revision"] == responder["revision"]
    assert qwen["tree_sha256"] == responder["tree_sha256"]
    assert qwen["files"] == responder["files"]
    assert qwen["bytes"] == responder["bytes"]
    assert qwen["license"] == responder["license"] == "Apache-2.0"
