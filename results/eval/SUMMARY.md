# Authoritative project evidence status

Generated: `2026-08-30T12:59:12.975675+00:00`

**Verdict:** `not_thesis_ready_evidence_gates_pending`. Thesis-ready: **false**.

This table is generated from `EVIDENCE_STATUS.json`. Component and synthetic proxies are never promoted to official end-to-end evidence.

## Acceptance gates

| Gate | Passed |
|---|---:|
| `audited_export_100_to_200_hours` | yes |
| `data_policy_resolved_under_documented_qa_waiver` | yes |
| `deployment_eligible_persian_direct_model` | no |
| `real_group_heldout_detector_above_80_percent` | no |
| `physical_12_to_24_gb_fit_and_live_latency` | no |
| `human_study_complete` | no |
| `source_code_license_selected` | no |

## Dataset and export

| Measure | Value |
|---|---:|
| Conversation source pairs | 6,754 |
| Conversation source hours | 123.796 |
| Audited exported pairs | 6,754 |
| Audited exported hours | 108.584 |
| Train / validation / test pairs | 6,419 / 131 / 204 |
| Session-group split leaks | 0 |
| Automatic integrity audit | pass |
| Human listening QA complete | no (waived) |

## Direct Moshi trials and ablations

| Trial | Context | LoRA rank | Embeddings trained | Candidates | Runtime passes per 9 | Minimum fixed-val loss | Deployment eligible |
|---|---:|---:|---|---:|---|---:|---:|
| v1 | 20.0 | 64 | upstream broad embedding switch | 16 | n/a | n/a | no |
| v2 | 100 | 64 | upstream broad embedding switch | 20 | 0, 0, 0, 2, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 | 1.734574 | no |
| v3 | 100.0 | 128 | none | 5 | 0, 0, 0, 1, 0 | 1.879258 | no |
| v4 | 100.0 | 128 | depformer_text_emb.weight, text_emb.weight | 5 | 1, 0, 0, 1, 0 | 1.879061 | no |

V1 was selected under its frozen loss protocol but failed later autoregressive runtime validation. V2–v4 each failed closed with zero eligible checkpoints; their frozen final tests remain untouched.

## Detector, latency, and study evidence

| Area | Current evidence | Official gate |
|---|---|---:|
| Detector | synthetic_proxy; synthetic n=32, accuracy=1.0 | pending |
| Hardware/latency | NVIDIA H100 NVL; official E2E rows=0 | pending |
| Human study | participants=1, aged 60+=1, complete ratings=0 | pending |
| Project license | not selected | pending |

## Remaining requirements

- Produce a validation-eligible Persian direct-model checkpoint before opening that experiment's frozen final test.
- Collect recorded, speaker/session-group-held-out interruption evidence.
- Run the final system and client-acknowledged latency protocol on a physical 12–24 GB GPU such as the planned RTX 4090.
- Collect consented complete results from 5–10 Persian speakers, including at least two aged 60+.
- Select and add the project source-code license.

No claim should exceed these evidence classes. In particular, the H100 measurements are engineering/training evidence, not physical-4090 official latency evidence, and synthetic detector accuracy is not a real held-out result.
