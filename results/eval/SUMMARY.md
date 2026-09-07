# Authoritative project evidence status

Generated: `2026-09-07T17:41:05.867358+00:00`

**Verdict:** `not_thesis_ready_evidence_gates_pending`. Thesis-ready: **false**.

This table is generated from `EVIDENCE_STATUS.json`. Component and synthetic proxies are never promoted to official end-to-end evidence.

## Acceptance gates

| Gate | Passed |
|---|---:|
| `working_persian_s2s_prototype` | yes |
| `automatic_semantic_final_test_passed` | yes |
| `audited_export_100_to_200_hours` | yes |
| `data_policy_resolved_under_documented_qa_waiver` | yes |
| `deployment_eligible_persian_direct_model` | no |
| `real_group_heldout_detector_above_80_percent` | no |
| `physical_12_to_24_gb_fit_and_live_latency` | no |
| `human_study_complete` | no |
| `source_code_license_selected` | yes |

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

## Working speech-to-speech system

The real-service cascade passed 9/9 fixed group-disjoint validation rows with 0 rule-fallback rows. Minimum reply Persian-script fraction was 1.0; minimum reply audio RMS was 0.133529. This is a positive working user-turn prototype result and out-of-sample mechanics check, not semantic or population-level generalization, human quality, physical-target, or official browser-latency evidence.

## Automatic semantic final test

The frozen Qwen3-4B prompt-v2 cascade result is verified and passed on 40 group-disjoint test rows with 240 valid judge calls and 0 failures. Mean relevance improved from 1.425 for the frozen 0.5B baseline to 3.15; candidate coherence was 3.325. Relevance gain was 1.725, coherence gain 1.8, paired relevance win rate 0.675, and relevance-at-least-two rate 0.875. Every predeclared automatic engineering gate passed. This is an automatic same-family LLM-as-judge proxy, not a human or independent benchmark; it does not establish factuality, safety, naturalness, population usefulness, physical-4090 fit, or browser latency.

## Privacy-safe descriptive error analysis

The hash-bound post-hoc analysis is verified. It found 9/9 automatic mechanics successes and 0 mechanics failures. Of the nine rows, 3 had transcripts over 1,000 characters, 3 had replies over 8 seconds, and 2 took over 10 seconds for complete generation. Transcript length and full-turn time had descriptive Pearson r=0.885282. This small, post-hoc association is not inferential, and full-turn time is not official first-audio latency. Plaintext was not read or emitted; this nine-row post-hoc analysis did not measure semantics. The separate frozen automatic semantic final test is reported above; pronunciation, naturalness, human quality, elderly performance, and population generalization remain unmeasured.

## Automatic synthesized-speech intelligibility proxy

The predeclared NeMo round-trip measurement is verified on 9 exact reproduced validation outputs with 0 ASR failures. Micro WER=0.304094 over 171 reference words; micro CER=0.071429 over 658 reference characters. No quality threshold was introduced after observation. This automatic single-voice ASR proxy does not establish human intelligibility, pronunciation, naturalness, semantic relevance, or population generalization.

## Duplex transport

The automated WebSocket regression passed with 4 tests. It verifies 3,200 pre-roll samples plus 1,600 continued samples become a 4,800-sample next-turn input, together with acknowledgement ingestion, identity telemetry, cancellation, reconnect, state isolation, error recovery, and metrics-only no-WAV retention. Actual browser source-stop, microphone, and physical-target evidence remain pending.

## Direct Moshi trials and ablations

| Trial | Scope | Context | Rank | Seed | Embeddings trained | Candidates | Val chunks/candidate | Runtime pass/fail per 9 | Min val loss | Eligible |
|---|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| v1 | heldout_loss_then_postselection_runtime | 20.0 | 64 | 20260823 | upstream broad embedding switch | 16 | n/a | n/a | n/a | no |
| v2 | validation_only_checkpoint_selection | 100 | 64 | 20260827 | upstream broad embedding switch | 20 | 202 | 0/9, 0/9, 0/9, 2/7, 1/8, 1/8, 1/8, 1/8, 0/9, 1/8, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9, 0/9 | 1.734574 | no |
| v3 | validation_only_checkpoint_selection | 100.0 | 128 | 20260827 | none | 5 | 202 | 0/9, 0/9, 0/9, 1/8, 0/9 | 1.879258 | no |
| v4 | validation_only_checkpoint_selection | 100.0 | 128 | 20260827 | depformer_text_emb.weight, text_emb.weight | 5 | 202 | 1/8, 0/9, 0/9, 1/8, 0/9 | 1.879061 | no |
| v5 | validation_only_checkpoint_selection | 100.0 | 128 | 20260827 | depformer_text_emb.weight, text_emb.weight | 5 | 202 | 0/9, 0/9, 0/9, 1/8, 1/8 | 1.867859 | no |
| v6.2 | train_only_in_sample_capacity_diagnostic | 12 | 64 | 20260901 | depformer_text_emb.weight, text_emb.weight, text_linear.frozen_W.weight | 4 | 32 | 0/9, 0/9, 0/9, 0/9 | 2.060297 | no |

V1 was selected under its frozen loss protocol but failed later autoregressive runtime validation. V2–v5 each failed closed with zero eligible checkpoints. V6.2 is a deliberately in-sample capacity diagnostic: its best post-step-50 text loss reduction was 32.20% but every checkpoint passed 0/9 direct-runtime rows. The v2–v6.2 final-test firewalls remain closed.

V1 automatic held-out loss evidence used 204 rows / 331 chunks: selected total-loss mean 1.7272041083230523 with 95% CI [1.5866204233506902, 1.8677877932954143]. This is automatic loss evidence only and does not repair the runtime failure or support a perceptual/deployment claim.

## Local artifact retention

The verified post-finalization cleanup reclaimed 41.01 GiB and retains 13 representative v1–v5 adapter tensors. Retained steps: v1=[500, 1000, 2000, 4000, 8000]; v2=[400, 2000]; v3=[400, 500]; v4=[400, 500]; v5=[400, 500].

V6.2 separately retains 4 current diagnostic checkpoints; they postdate that cleanup receipt.

All retained adapter hashes match committed evidence, and all scientific results, configurations, runtime outputs, and certificates remain. The full set of negative intermediate tensors is intentionally not retained; exact tensor recreation would require rerunning the frozen training recipes. Historical candidate counts and verdicts remain attested by their committed certificates.

## Detector, latency, and study evidence

| Area | Current evidence | Official gate |
|---|---|---:|
| Detector | synthetic accuracy=1.0; recorded proxy n=132 / 22 sessions, accuracy=0.8106060606060606, F1=0.7899159663865547, session CI=[0.7444, 0.8718]; automatic labels, not official ground truth | pending |
| Hardware/latency | NVIDIA H100 NVL; official E2E rows=0 | pending |
| Human study | participants=1, aged 60+=1, complete ratings=0 | pending |
| Project license | LICENSE | pass |

## Reporting contract

Denominators and observed failures are shown above. Model seeds are reported per trial; the Moshi data split is frozen by `source_session_id`. Confidence intervals are reported where estimable, including event and session-block intervals for the recorded automatic-label proxy. Official latency, independently labeled detector, and human-study intervals remain null because their qualifying denominators are zero. Exact timing and acceptance definitions are in `docs/METRICS.md`.

## Submission architecture decision

Production candidate: `qwen4b_v2_cascade`. Direct Moshi role: `experimental_negative_result_with_positive_learning_signal`. Starting another direct-model training run before the deadline is not recommended because no validated corrective hypothesis remains, while the prompt-v2 cascade has passed both its eligibility stage and frozen automatic semantic final test.

## Remaining requirements

- Produce a validation-eligible Persian direct-model checkpoint before opening that experiment's frozen final test.
- Obtain independently human-reviewed recorded labels and establish >80% on a speaker/session-group-held-out test; the automatic YouTube proxy is not ground truth.
- Run the final system and client-acknowledged latency protocol on a physical 12–24 GB GPU such as the planned RTX 4090.
- Collect consented complete results from 5–10 Persian speakers, including at least two aged 60+.

No claim should exceed these evidence classes. In particular, the H100 measurements are engineering/training evidence, not physical-4090 official latency evidence; synthetic accuracy and the recorded automatic-label proxy are not independently labeled official detector evidence.
