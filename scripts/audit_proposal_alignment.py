#!/usr/bin/env python3
"""Fail-closed alignment audit for the detailed thesis proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PROPOSAL_SHA256 = (
    "658a75f01f0d25cb18e601f8f5d36438c0c3780666a0f95c8fe529a6d68f90b4"
)
EXPECTED_PROPOSAL_NAME = "Thesis Proposal Template.docx"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_report(
    proposal_path: Path,
    evidence: dict[str, Any],
    *,
    expected_proposal_sha256: str = EXPECTED_PROPOSAL_SHA256,
    expected_proposal_name: str = EXPECTED_PROPOSAL_NAME,
    evidence_sha256: str,
    evidence_path: str = "results/eval/EVIDENCE_STATUS.json",
    verified_proposal_sha256: str | None = None,
) -> dict[str, Any]:
    observed_hash = verified_proposal_sha256 or sha256_file(proposal_path)
    if observed_hash != expected_proposal_sha256:
        raise ValueError("proposal SHA-256 drift; review and remap requirements before rerunning")
    if proposal_path.name != expected_proposal_name:
        raise ValueError("proposal filename drift; review and remap requirements before rerunning")
    if len(evidence_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in evidence_sha256
    ):
        raise ValueError("evidence_sha256 must be a lowercase SHA-256 digest")

    gates = evidence.get("gates") or {}
    data = evidence.get("data") or {}
    direct = evidence.get("direct_moshi") or {}
    duplex = evidence.get("duplex_transport") or {}
    hardware = evidence.get("latency_and_hardware") or {}
    study = evidence.get("human_study") or {}
    recorded_proxy = (evidence.get("detector") or {}).get("recorded_proxy") or {}

    exported_hours = float(data.get("exported_hours") or 0.0)
    data_substantial = bool(
        100.0 <= exported_hours <= 200.0
        and int(data.get("group_split_leaks") or 0) == 0
        and not (data.get("audit_failure_counts") or {})
    )
    data_strict = bool(
        data_substantial
        and data.get("strict_human_qa_complete")
        and data.get("human_verified_claim_allowed")
    )
    direct_trials = direct.get("trials") or []
    direct_substantial = bool(
        direct_trials and any(row.get("scientific_run_complete") for row in direct_trials)
    )
    direct_strict = bool(direct.get("deployment_eligible") and direct.get("promoted_adapter"))
    streaming_substantial = bool(duplex.get("automated_websocket_regression_passed"))
    streaming_strict = bool(
        gates.get("physical_12_to_24_gb_fit_and_live_latency")
        and hardware.get("official_e2e_rows")
        and hardware.get("official_latency_gate_passed")
    )
    duplex_substantial = bool(
        streaming_substantial and recorded_proxy.get("present")
    )
    duplex_strict = bool(
        gates.get("real_group_heldout_detector_above_80_percent")
        and duplex.get("official_full_duplex_evidence")
        and duplex.get("official_interrupt_latency_le_150_ms")
    )
    human_substantial = bool(study.get("complete_ratings"))
    human_strict = bool(
        gates.get("human_study_complete")
        and 5 <= int(study.get("participants") or 0) <= 10
        and int(study.get("elderly_participants") or 0) >= 2
        and float(study.get("mos_mean") or 0.0) >= 3.5
    )

    criteria = [
        {
            "id": "dataset_100_to_200_annotated_hours",
            "proposal_criterion": ">=100 hours with turn boundaries and overlap labels",
            "substantially_implemented": data_substantial,
            "strictly_passed": data_strict,
            "observed": {
                "exported_hours": exported_hours,
                "pairs": int(data.get("exported_pairs") or 0),
                "group_split_leaks": int(data.get("group_split_leaks") or 0),
                "strict_human_qa_complete": bool(data.get("strict_human_qa_complete")),
            },
            "gap": "Quantity and machine integrity pass; turn/overlap labels are automatic and listening QA is waived.",
        },
        {
            "id": "direct_s2s_persian_finetune",
            "proposal_criterion": "fine-tuned direct S2S model with reduced loss and intelligible Persian output",
            "substantially_implemented": direct_substantial,
            "strictly_passed": direct_strict,
            "observed": {
                "trials": len(direct_trials),
                "deployment_eligible": bool(direct.get("deployment_eligible")),
                "positive_learning_trial": any(
                    row.get("positive_learning_result") for row in direct_trials
                ),
            },
            "gap": "Moshika training produced a positive loss signal, but every direct runtime candidate failed and no adapter was promoted.",
        },
        {
            "id": "streaming_latency_500_ms",
            "proposal_criterion": "qualifying <=500 ms end-to-end latency on one 12-24 GB GPU",
            "substantially_implemented": streaming_substantial,
            "strictly_passed": streaming_strict,
            "observed": {
                "transport_regression_passed": streaming_substantial,
                "physical_target_ready": bool(hardware.get("physical_target_hardware_ready")),
                "official_e2e_rows": int(hardware.get("official_e2e_rows") or 0),
            },
            "gap": "Streaming transport exists, but no physical target/browser rows establish mean, p90, or maximum latency.",
        },
        {
            "id": "full_duplex_interrupt_80_f1_150_ms",
            "proposal_criterion": ">=0.80 interrupt F1 plus <=150 ms stop/adaptation behavior",
            "substantially_implemented": duplex_substantial,
            "strictly_passed": duplex_strict,
            "observed": {
                "automatic_proxy_accuracy": recorded_proxy.get("accuracy"),
                "automatic_proxy_interrupt_f1": recorded_proxy.get("interrupt_f1"),
                "human_verified_labels": int(recorded_proxy.get("human_verified_labels") or 0),
                "official_full_duplex_evidence": bool(duplex.get("official_full_duplex_evidence")),
            },
            "gap": "The automatic-label proxy has 0.8106 accuracy but only 0.7899 F1; independent labels and physical <=150 ms traces are absent.",
        },
        {
            "id": "human_mos_3_5_elderly",
            "proposal_criterion": "MOS >=3.5 from 5-10 Persian speakers including >=2 aged 60+",
            "substantially_implemented": human_substantial,
            "strictly_passed": human_strict,
            "observed": {
                "participants": int(study.get("participants") or 0),
                "elderly_participants": int(study.get("elderly_participants") or 0),
                "complete_ratings": int(study.get("complete_ratings") or 0),
                "mos_mean": study.get("mos_mean"),
            },
            "gap": "Study tooling exists, but there are zero complete ratings and no qualifying MOS or elderly finding.",
        },
    ]
    traced = len(criteria)
    substantial = sum(bool(row["substantially_implemented"]) for row in criteria)
    passed = sum(bool(row["strictly_passed"]) for row in criteria)
    return {
        "schema_version": 2,
        "evidence_class": "proposal_requirements_alignment_audit",
        "proposal": {
            "path": proposal_path.name,
            "sha256": observed_hash,
            "hash_verified": True,
            "document_status": "unsigned_template_with_unfilled_identity_and_signature_fields",
        },
        "aggregate_evidence": {
            "path": evidence_path,
            "sha256": evidence_sha256,
            "schema_version": evidence.get("schema_version"),
        },
        "scoring_contract": {
            "criteria": traced,
            "traceability_percent": round(100.0 * traced / 5, 1),
            "substantial_implementation_percent": round(100.0 * substantial / 5, 1),
            "strict_acceptance_percent": round(100.0 * passed / 5, 1),
            "warning": "Implementation coverage is not acceptance. A criterion passes only with its proposal-level evidence.",
        },
        "criteria": criteria,
        "proposal_internal_ambiguities": [
            "The problem statement binds p90 latency, the objective table binds average latency, and neither defines the timing endpoints.",
            "The detector objective says accuracy >=80% while labeling the criterion F1; accuracy and F1 are not interchangeable.",
            "The final note allows an undergraduate project to prioritize latency or full duplex, but no filled or signed scope choice is present.",
        ],
        "architecture_mismatch": {
            "proposal_requires_no_intermediate_text": True,
            "positive_production_candidate_is_cascade": True,
            "direct_branch_deployment_eligible": bool(direct.get("deployment_eligible")),
            "strictly_aligned": False,
        },
        "verdict": "directionally_aligned_but_not_strictly_requirement_complete",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, default=ROOT / "Thesis Proposal Template.docx")
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "results/eval/EVIDENCE_STATUS.json"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/proposal_alignment_audit.json"
    )
    args = parser.parse_args()
    evidence_bytes = args.evidence.read_bytes()
    evidence = json.loads(evidence_bytes.decode("utf-8"))
    try:
        evidence_path = args.evidence.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        evidence_path = args.evidence.name
    report = build_report(
        args.proposal,
        evidence,
        evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        evidence_path=evidence_path,
    )
    _write_json(args.out, report)
    print(json.dumps(report["scoring_contract"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
