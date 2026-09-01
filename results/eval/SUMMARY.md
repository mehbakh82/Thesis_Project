# Authoritative project evidence status

Generated: `2026-09-01T05:52:29.292464+00:00`

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
| Machine audit failures | 0 |
| Split policy | source-session-group isolated |
| Automatic integrity audit | pass |
| Human listening QA complete | no (waived) |

## Direct Moshi trials and ablations

| Trial | Context | Rank | Seed | Embeddings trained | Candidates | Val chunks/candidate | Runtime pass/fail per 9 | Min val loss | Eligible |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| v1 | 20.0 | 64 | 20260823 | upstream broad embedding switch | 16 | n/a | n/a | n/a | no |
| v2 | 100 | 64 | 20260827 | upstream broad embedding switch | 20 | 202 | 0/9, 0/9, 0/9, 2/7, 1/8, 1/8, 1/8, 1/8, 0/9, 1/8, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9 | 1.734574 | no |
| v3 | 100.0 | 128 | 20260827 | none | 5 | 202 | 0/9, 0/9, 0/9, 1/8, 0/9 | 1.879258 | no |
| v4 | 100.0 | 128 | 20260827 | depformer_text_emb.weight, text_emb.weight | 5 | 202 | 1/8, 0/9, 0/9, 1/8, 0/9 | 1.879061 | no |
| v5 | 100.0 | 128 | 20260827 | depformer_text_emb.weight, text_emb.weight | 5 | 202 | 0/9, 0/9, 0/9, 1/8, 1/8 | 1.867859 | no |

V1 was selected under its frozen loss protocol but failed later autoregressive runtime validation. V2–v5 each failed closed with zero eligible checkpoints; their frozen final tests remain untouched.

V1 automatic held-out loss evidence used 204 rows / 331 chunks: selected total-loss mean 1.7272041083230523 with 95% CI [1.5866204233506902, 1.8677877932954143]. This is automatic loss evidence only and does not repair the runtime failure or support a perceptual/deployment claim.

## Local artifact retention

The verified post-finalization cleanup reclaimed 41.01 GiB and retains 13 representative adapter tensors. Retained steps: v1=[500, 1000, 2000, 4000, 8000]; v2=[400, 2000]; v3=[400, 500]; v4=[400, 500]; v5=[400, 500].

All retained adapter hashes match committed evidence, and all scientific results, configurations, runtime outputs, and certificates remain. The full set of negative intermediate tensors is intentionally not retained; exact tensor recreation would require rerunning the frozen training recipes. Historical candidate counts and verdicts remain attested by their committed certificates.

## Detector, latency, and study evidence

| Area | Current evidence | Official gate |
|---|---|---:|
| Detector | synthetic accuracy=1.0; recorded proxy n=132 / 22 sessions, accuracy=0.8106060606060606, F1=0.7899159663865547, session CI=[0.7444, 0.8718]; automatic labels, not official ground truth | pending |
| Hardware/latency | NVIDIA H100 NVL; official E2E rows=0 | pending |
| Human study | participants=1, aged 60+=1, complete ratings=0 | pending |
| Project license | not selected | pending |

## Reporting contract

Denominators and observed failures are shown above. Model seeds are reported per trial; the Moshi data split is frozen by `source_session_id`. Confidence intervals are reported where estimable, including event and session-block intervals for the recorded automatic-label proxy. Official latency, independently labeled detector, and human-study intervals remain null because their qualifying denominators are zero. Exact timing and acceptance definitions are in `docs/METRICS.md`.

## Remaining requirements

- Produce a validation-eligible Persian direct-model checkpoint before opening that experiment's frozen final test.
- Obtain independently human-reviewed recorded labels and establish >80% on a speaker/session-group-held-out test; the automatic YouTube proxy is not ground truth.
- Run the final system and client-acknowledged latency protocol on a physical 12–24 GB GPU such as the planned RTX 4090.
- Collect consented complete results from 5–10 Persian speakers, including at least two aged 60+.
- Select and add the project source-code license.

No claim should exceed these evidence classes. In particular, the H100 measurements are engineering/training evidence, not physical-4090 official latency evidence; synthetic accuracy and the recorded automatic-label proxy are not independently labeled official detector evidence.
