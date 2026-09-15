import hashlib
from pathlib import Path

from scripts.audit_proposal_alignment import build_report


def test_proposal_alignment_separates_implementation_from_acceptance(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.docx"
    proposal.write_bytes(b"fixed proposal fixture")
    expected = hashlib.sha256(proposal.read_bytes()).hexdigest()
    evidence = {
        "gates": {
            "physical_12_to_24_gb_fit_and_live_latency": False,
            "real_group_heldout_detector_above_80_percent": False,
            "human_study_complete": False,
        },
        "data": {
            "exported_hours": 108.584,
            "exported_pairs": 6754,
            "group_split_leaks": 0,
            "audit_failure_counts": {},
            "strict_human_qa_complete": False,
            "human_verified_claim_allowed": False,
        },
        "direct_moshi": {
            "deployment_eligible": False,
            "promoted_adapter": None,
            "trials": [
                {"scientific_run_complete": True, "positive_learning_result": True}
            ],
        },
        "duplex_transport": {
            "automated_websocket_regression_passed": True,
            "official_full_duplex_evidence": False,
        },
        "detector": {
            "recorded_proxy": {
                "present": True,
                "accuracy": 0.8106,
                "interrupt_f1": 0.7899,
                "human_verified_labels": 0,
            }
        },
        "latency_and_hardware": {
            "physical_target_hardware_ready": False,
            "official_e2e_rows": 0,
        },
        "human_study": {
            "participants": 1,
            "elderly_participants": 1,
            "complete_ratings": 0,
        },
    }

    report = build_report(
        proposal,
        evidence,
        expected_proposal_sha256=expected,
        expected_proposal_name=proposal.name,
        evidence_sha256="a" * 64,
    )

    assert report["schema_version"] == 2
    assert report["aggregate_evidence"] == {
        "path": "results/eval/EVIDENCE_STATUS.json",
        "sha256": "a" * 64,
        "schema_version": None,
    }
    assert report["scoring_contract"]["traceability_percent"] == 100.0
    assert report["scoring_contract"]["substantial_implementation_percent"] == 80.0
    assert report["scoring_contract"]["strict_acceptance_percent"] == 0.0
    assert report["architecture_mismatch"]["strictly_aligned"] is False


def test_proposal_latency_duplex_and_human_criteria_are_closable(tmp_path: Path) -> None:
    proposal = tmp_path / "proposal.docx"
    proposal.write_bytes(b"fixed proposal fixture")
    expected = hashlib.sha256(proposal.read_bytes()).hexdigest()
    evidence = {
        "gates": {
            "physical_12_to_24_gb_fit_and_live_latency": True,
            "real_group_heldout_detector_above_80_percent": True,
            "human_study_complete": True,
        },
        "duplex_transport": {
            "automated_websocket_regression_passed": True,
            "official_full_duplex_evidence": True,
            "official_interrupt_latency_le_150_ms": True,
        },
        "detector": {
            "recorded_proxy": {
                "present": True,
                "accuracy": 0.84,
                "interrupt_f1": 0.83,
                "human_verified_labels": 30,
            }
        },
        "latency_and_hardware": {
            "physical_target_hardware_ready": True,
            "official_e2e_rows": 10,
            "official_latency_gate_passed": True,
        },
        "human_study": {
            "participants": 5,
            "elderly_participants": 2,
            "complete_ratings": 5,
            "mos_mean": 3.5,
        },
    }

    report = build_report(
        proposal,
        evidence,
        expected_proposal_sha256=expected,
        expected_proposal_name=proposal.name,
        evidence_sha256="b" * 64,
    )
    criteria = {row["id"]: row for row in report["criteria"]}

    assert criteria["streaming_latency_500_ms"]["strictly_passed"] is True
    assert criteria["full_duplex_interrupt_80_f1_150_ms"]["strictly_passed"] is True
    assert criteria["human_mos_3_5_elderly"]["strictly_passed"] is True
    assert report["scoring_contract"]["strict_acceptance_percent"] == 60.0
