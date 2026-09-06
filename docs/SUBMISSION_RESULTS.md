# Submission results and claim boundary

Status date: 2026-09-06

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

## Privacy-safe descriptive error analysis

The canonical post-hoc artifact is
`results/eval/cascade_validation_descriptive_analysis.json`, hash-bound to the
unchanged validation report (`ae515381db2b72e9ed0b8dcb491c0765990732ed8bb8c46383499b9d60f89ced`).
It processed only the report's aggregate measurements and hashes; it did not
read or emit transcript/reply plaintext and did not access the frozen final
test.

| Observation | Result |
|---|---:|
| Automatic mechanics successes / failures | **9 / 0** |
| ASR errors / responder fallbacks / language retries | 0 / 0 / 0 |
| Unique transcript / reply hashes | 9 / 9 |
| Transcript characters p50 / p95 / max | 445 / 3977.8 / 5031 |
| Reply characters p50 / p95 / max | 110 / 115.6 / 116 |
| Reply-audio seconds p50 / p95 / max | 7.465 / 9.629 / 9.857 |
| Rows with transcript >1000 characters | 3/9 |
| Rows with reply audio >8 seconds | 3/9 |
| Rows with complete generation >10 seconds | 2/9 |
| Transcript-length/full-turn Pearson correlation | 0.885 (descriptive) |

No automatic mechanics failure occurred in this small panel. The principal
observed engineering risk is instead long complete-response time, especially
for long ASR transcripts: the longest transcript and slowest turn were both
manifest row 32 (5031 characters; 13270.249 ms). This is a post-hoc descriptive
association over nine cases, not an inferential result or an official
`T_first_audio` measurement. Because plaintext and audio were deliberately not
reviewed, this analysis cannot assess semantic relevance, ASR correctness,
pronunciation, naturalness, human quality, elderly performance, or population
generalization.

## Automatic synthesized-speech intelligibility proxy

The protocol in `docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md` was committed and
pushed at `3bb3a85` before the measurement. It bound the unchanged validation
report and manifest, the same nine indices, exact Qwen revision, Piper model,
normalization, CER/WER definitions, validity gates, privacy policy, and absence
of a quality threshold. The resulting
`results/eval/cascade_intelligibility_proxy.json` reproduced every canonical
input-transcript and reply hash and completed both ASR calls on all nine rows.

| Automatic round-trip measure | Result |
|---|---:|
| Valid rows / ASR failures | **9 / 0** |
| Reference words / word edits | 171 / 52 |
| Micro / macro-mean WER | **30.41% / 32.01%** |
| WER p50 / p95 / max | 28.00% / 49.40% / 54.55% |
| Reference characters / character edits | 658 / 47 |
| Micro / macro-mean CER | **7.14% / 7.13%** |
| CER p50 / p95 / max | 7.50% / 9.40% / 9.62% |

This supports only an automatic single-voice content-preservation proxy. The
same ASR family is used in the system and measurement, Persian spacing makes
WER substantially harsher than CER, and nine outputs cannot establish human
intelligibility, pronunciation, naturalness, semantic quality, or population
generalization. No confidence interval or after-the-fact passing threshold is
reported.

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
| Cascade descriptive analysis | `results/eval/cascade_validation_descriptive_analysis.json` |
| Automatic cascade intelligibility proxy | `results/eval/cascade_intelligibility_proxy.json` |
| Frozen intelligibility protocol | `docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md` |
| Frozen cascade protocol | `docs/CASCADE_VALIDATION_PROTOCOL.md` |
| Duplex transport continuity | `results/release/final_audit.json` |
| V6.2 training integrity | `results/hardware/moshi_v6_text_dropout_training.json` |
| V6.2 losses | `results/moshi_v6_text_dropout_reevaluation.json` |
| V6.2 runtime failure | `results/moshi_v6_text_dropout_runtime.json` |
| Corpus/export | `results/conversation_audit.json`, `results/moshi_export_audit.json` |
| Detector proxy | `results/eval/interrupt_recorded_proxy.json` |
| Aggregate status | `results/eval/EVIDENCE_STATUS.json`, `results/eval/SUMMARY.md` |

## Deadline handoff

The Persian-reporting pre-license evidence freeze at `a0c4fcc` is pushed to
the private GitHub repository. GitHub Actions run `34042698214` passed every
gate, and an independent fresh full-history clone passed the tracked
artifact/history/privacy audit and all 183 tests. The snapshot binds 310
reproducibility-critical files, including the Persian reporting handoff. The
ignored private validation manifest was absent, as intended. This establishes
repository reproducibility; it does not change any scientific acceptance
result above.

A copy-ready Persian abstract, methods/results tables, discussion, limitations,
and conclusion—bounded to these exact evidence classes—are provided in
`docs/THESIS_REPORTING_FA.md`.

1. Copy the two result tables and safe wording above into the thesis; update
   abstract, methods, results, discussion, limitations, and conclusion.
2. Choose a source-code license.
3. The current pre-license repository freeze, tag, CI, and fresh-clone
   verification are complete. Regenerate them only if a license, tracked
   manuscript artifact, physical-target result, or human result is later added.
4. Run the physical-target/browser protocol only if suitable hardware arrives
   before the freeze. Otherwise report it as unavailable future work.

Human study, independent detector labels, waived listening QA, and a redesigned
direct model are strict-rubric research extensions. They are not safely
completable in code within the remaining time and must stay explicit
limitations.
