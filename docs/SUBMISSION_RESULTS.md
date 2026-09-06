# Submission results and claim boundary

Status date: 2026-09-05

## Final architecture decision

Use the modular cascade as the submitted working system:

```text
continuous user microphone
    -> classical rolling barge-in detector and browser stop control
    -> NeMo Persian ASR
    -> exact local Qwen2.5-0.5B-Instruct
    -> Persian Piper TTS
```

At the WebSocket transport level, an automatic or manual barge-in now carries
the detector's 450 ms rolling microphone pre-roll into continued capture, so
the interruption becomes the next user turn. Regression tests also cover
acknowledgement persistence, cancellation, simultaneous turns, reconnect,
malformed/silent input, recoverable generation/OOM failure, identity telemetry,
and metrics-only no-WAV retention. These tests do not replace a physical
browser/audio-device trace.

The direct Moshika/Moshi branch is an experimental research result, not the
production path. Do not start another speculative direct training run before
the deadline. V6.2 supplies a useful positive learning signal and a clear
negative generation result, while the cascade already has a frozen positive
validation result.

## Primary positive result

The canonical artifact is
`results/eval/cascade_real_service_validation_panel.json`.

| Measure | Result |
|---|---:|
| Validation manifest | 131 rows |
| Split unit | `source_session_id` |
| Split leaks | 0 |
| Predeclared evaluated rows | 9 |
| Rows passing every automatic gate | **9/9** |
| Audited ASR input channel | user channel 1 |
| NeMo health | HTTP 200 |
| Qwen initialization/generation failures | 0/9 |
| Rule-fallback rows | **0/9** |
| Language-retry rows | 0/9 |
| Minimum transcript Persian-script fraction | 1.000 |
| Minimum reply Persian-script fraction | 1.000 |
| Non-silent Piper replies | **9/9** |
| Minimum reply RMS | 0.133529 |
| Descriptive full-turn time p50 / p95 / max | 5584.736 / 13045.919 / 13270.249 ms |
| Final-test rows accessed | 0 |

This is a positive working user-turn result and a group-disjoint,
out-of-sample mechanics check. The timing values measure complete
ASR-plus-response-plus-full-TTS generation on an H100; they are not streaming
browser `T_first_audio` and cannot satisfy the 500 ms requirement.

The prior cascade v4 development panel also passed 9/9, but its evaluator used
assistant channel 0 as ASR input. That result remains a transparent
component-chain smoke test. The issue was found and corrected to audited user
channel 1 before any validation row was read; the validation protocol and
evaluator were committed at `b0366e7` before the one-time run.

## Direct-model result

V6.2 was a deliberately train-only 32-row capacity diagnostic with rank-64
LoRA, full Persian text embeddings/output head, audio-loss weight 0.1, and
scheduled text-input dropout from 0.25 to 0.75. It is not a validation or
generalization experiment.

| Step | Total loss | Text loss | Audio loss | Direct runtime |
|---:|---:|---:|---:|---:|
| 50 | 2.636405 | 0.812474 | 1.823931 | 0/9 |
| 100 | 2.245344 | 0.638987 | 1.606357 | 0/9 |
| 150 | 2.071542 | **0.550852** | 1.520690 | 0/9 |
| 200 | **2.060297** | 0.551183 | **1.509114** | 0/9 |

The best post-step-50 text loss is 32.20% lower than the step-50 value and
passes the frozen 30% capacity criterion. Nevertheless, every checkpoint
passes 0/9 official-server output rows; only 1/9 rows per checkpoint emits both
text and speech. The correct conclusion is:

> V6.2 learned the bounded in-sample training objective, but the learned
> behavior did not transfer to reliable autoregressive Persian
> speech-and-text generation.

No v6.2 checkpoint is selected or deployment-eligible. No final test was
opened.

## Safe thesis wording

Recommended:

> We implemented a Persian full-duplex speech prototype using a modular
> NeMo–Qwen–Piper cascade and a continuous-microphone barge-in controller. In a
> frozen nine-item panel drawn from a source-session-group-isolated validation
> split, the real component chain passed all automatic execution, Persian
> script, no-fallback, and non-silent-audio gates. Direct Moshika adaptation
> showed measurable in-sample objective learning but failed the predeclared
> autoregressive runtime criterion, motivating use of the cascade as the final
> prototype and preserving the direct-model result as a negative ablation.

Do not claim:

- that direct Moshi works in Persian;
- that the nine-row cascade panel proves semantic response quality,
  naturalness, or population-level generalization;
- that full-turn H100 time is browser first-audio latency;
- that the system satisfies the physical 12–24 GB or ≤500 ms gate;
- that automatic interaction labels are independently human verified;
- that the human study or elderly evaluation is complete;
- that waived listening QA was performed.

## Evidence map

| Claim | Canonical evidence |
|---|---|
| Working cascade | `results/eval/cascade_real_service_validation_panel.json` |
| Frozen cascade protocol | `docs/CASCADE_VALIDATION_PROTOCOL.md` |
| Duplex transport continuity | `results/release/final_audit.json` |
| V6.2 training integrity | `results/hardware/moshi_v6_text_dropout_training.json` |
| V6.2 losses | `results/moshi_v6_text_dropout_reevaluation.json` |
| V6.2 runtime failure | `results/moshi_v6_text_dropout_runtime.json` |
| Corpus/export | `results/conversation_audit.json`, `results/moshi_export_audit.json` |
| Detector proxy | `results/eval/interrupt_recorded_proxy.json` |
| Aggregate status | `results/eval/EVIDENCE_STATUS.json`, `results/eval/SUMMARY.md` |

## Remaining two-day path

1. Copy the two result tables and safe wording above into the thesis; update
   abstract, methods, results, discussion, limitations, and conclusion.
2. Choose a source-code license.
3. Complete the final repository audit and regenerate the release snapshot.
4. Freeze a submission commit and tag, push both, and verify a clean
   full-history clone.
5. Run the physical-target/browser protocol only if suitable hardware arrives
   before the freeze. Otherwise report it as unavailable future work.

Human study, independent detector labels, waived listening QA, and a redesigned
direct model are strict-rubric research extensions. They are not safely
completable in code within the remaining time and must stay explicit
limitations.
