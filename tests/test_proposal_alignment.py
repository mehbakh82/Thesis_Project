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

    report = build_report(proposal, evidence, expected_proposal_sha256=expected)

    assert report["scoring_contract"]["traceability_percent"] == 100.0
    assert report["scoring_contract"]["substantial_implementation_percent"] == 80.0
    assert report["scoring_contract"]["strict_acceptance_percent"] == 0.0
    assert report["architecture_mismatch"]["strictly_aligned"] is False
