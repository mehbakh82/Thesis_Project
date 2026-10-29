# Semantic end-of-turn gate v2 — result

Protocol: `docs/END_OF_TURN_V2_PROTOCOL.md`. Receipts:
`results/eval/end_of_turn_v2_dev.json` (development selection),
`results/eval/live_loopback_v3.json` (loopback test) and
`results/eval/live_loopback_v3_splits.json` (split utterances against v2).
**Outcome: the selected gate passed development but failed the loopback
latency criteria, so it is removed. The deployed endpointer stays unchanged
(256 ms confirmation, 96 ms speculative start).**

## Clip-end labels

Gemini labelled the 425 development clips in two passes with different
wording:

| Label | Clips |
|---|---:|
| Complete (both passes) | 158 |
| Incomplete (both passes) | 224 |
| Uncertain (passes disagree) | 43 |

The passes agreed on 90% of clips. More than half of the clips end
mid-sentence, far more than the 13% estimated from the first run's lexical
inspection, because most 1–5 s clips are slices of longer podcast turns. False
holds are now counted only on the 316 trials whose clip end is labelled
complete (both conditions).

## Development selection

| Candidate (best configuration) | Premature ends | False holds on complete ends | v1 false-hold measure | Holds of unfinished ends |
|---|---:|---:|---:|---:|
| none (deployed) | 19.8% | 0% | 0% | 0% |
| L — lexical rule, H 1024 | 14.4% | 1.6% | 13.2% | 23% |
| **B — Qwen3-4B completeness, θ 0.3, H 640** | **13.3%** | **4.4%** | 15.4% | 24% |
| A — Smart Turn v3.2 zero-shot, θ 0.3 | 15.2% | 16.5% | 25.8% | 33% |
| Stage 2 (v1) — Smart Turn, Persian fine-tune | 2.4% | 41.5% | 65.3% | 84% |
| Stage 2b — Smart Turn, restored windows | 4.2% | 29.1% | 56.5% | 77% |

Under the rule, B with θ 0.3 and H 640 ms is chosen:

- It lowers premature ends from 19.8% to 13.3%, a 33% relative reduction
  (plain 78 → 52 and simulated elderly 90 → 61 of 425 trials each).
- It holds 4.4% of complete ends, within the 5% limit.
- It adds 59 ms on average, and no answer arrived too late.

The lexical rule also qualifies, but its premature-end rate (14.4%) is 1.5
points above the best eligible configuration (B with H 1024, 12.9%), outside
the one-point tie band. Within the band, the smaller hold (640 ms) wins.

### Stage 2b

- **Mining:** the restored windows (1,068 training-split windows, 233 h, no
  development or test session) gave 7,905 complete and 22,169 incomplete
  examples.
- **Label filter:** Gemini agreed with the diarization label for only 65% of
  complete and 69% of incomplete examples. Diarization boundaries are noisy
  labels for turn ends.
- **Deviation:** the shared Gemini gateway budget ran out during the filter.
  15,880 of 29,563 transcribed examples were labelled, from 94 sessions and all
  four channels. The rest were dropped, as the protocol specifies for failed
  batches. The balanced set therefore has 2,749 examples per class (4,952
  training, 546 validation) instead of up to 12,000.
- **Acoustics:** both classes are cut where the live voice detector ends
  speech. Complete examples end in a natural fade (last 60 ms at −28 dB
  relative to the clip, against −11 dB for incomplete ones) and are longer
  (median 8.0 s against 6.4 s).
- **Training:** validation accuracy rose from 0.65 to 0.87 (best epoch 2).
  The ONNX export matches PyTorch exactly.
- **Development trials:** the model still held 29–45% of complete ends,
  better than the v1 fine-tune but far from the 5% limit. The training clips
  are longer than the 1–5 s development clips, which may push the model
  towards holding short utterances.

## Loopback test (v3)

Plain (201) and simulated-elderly (80) panels; same harness, host, stack and
scorer as v2; the gate was the only change.

| Measure | v2 (no gate) | v3 (gate) | Criterion |
|---|---:|---:|---|
| Split utterances, pooled | 22 | 17 | lower than v2 ✓ (McNemar p = 0.30) |
| Plain: split / mean / p90 | 15 / 372 / 431 ms | 11 / 363 / 656 ms | mean ✓, p90 ≤ 500 ✗ |
| Elderly: split / p90 | 7 / 446 ms | 6 / 656 ms | p90 ≤ 500 ✗ |
| Plain turns answered within 500 ms | 100% | 87.6% | — |
| Replies missing | 0 | 0 | — |

The server telemetry shows how it happened:

- The gate held 78 of 358 pauses. In 34 the speaker went on, so the hold was
  correct and avoided a split.
- The other 44 holds fell on finished turns: 14.8% of the 298 answered turns.
- A hold delays the reply by up to H − 256 = 384 ms. Held turns had a median
  first audio of 671 ms, against 318 ms for turns that were not held.
- So more than a tenth of turns were slow, which pushed the 90th percentile to
  656 ms. The rule requires all three conditions,
so the gate is removed. The difference in split utterances (22 → 17) is not
significant either. The stack peaked at 12,016 MiB, the same budget as v2.

## What stays

- **Deployed endpointer:** unchanged. Splitting on mid-sentence pauses stays a
  reported failure mode: 15 of 201 plain and 7 of 80 simulated-elderly
  utterances split in v2.
- **Gate code:** the gate mechanism and all candidates, including the stage-2b
  model (`models/turn/smart-turn-v3.2-fa-v2.onnx`, not in the repository),
  stay in the code, off by default.
- **Lesson:** a gate whose holds cost 384 ms needs a false-hold rate low enough
  on real turn ends that fewer than a tenth of turns are held. On spontaneous
  Persian speech the zero-shot gates hold too many finished turns. A shorter
  hold, or a gate trained on human-labelled turn ends, would be the next step.

## Claim boundary

Automatic clip-end labels from one labeller family; offline development
replay; automated loopback with recorded and tempo-slowed speech on the shared
H100 host; simulated elderly speech.
