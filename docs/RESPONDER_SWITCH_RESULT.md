# Responder switch — result

Protocol: `docs/RESPONDER_SWITCH_PROTOCOL.md`. The deployed responder changes
to Qwen3.5-4B with elderly prompt v1 only if **both** parts pass.

## Part 2 — deployment qualification (measured, 2026-10-20)

Receipts: `results/eval/responder_switch_qualification_loopback.json`,
`results/eval/responder_switch_qualification_coherence.json`, combined in
`results/eval/responder_switch_qualification.json`. Same panels, harness, host
and scorer as live loopback v2; the final endpointer configuration (no
end-of-turn gate was adopted, `docs/END_OF_TURN_RESULT.md`); Qwen3.5-4B bf16 in
vLLM at a 0.11 GPU-memory share, thinking disabled, Persian-only decoding, at
most 96 new tokens.

| Check | Qwen3.5-4B, elderly prompt | Deployed Qwen3-4B (v2) | Limit | Result |
|---|---|---|---|---|
| Q1 peak GPU memory, whole stack | 11,326 MiB | 11,336 MiB | 12,288 MiB | pass |
| Q2 plain (201): missing / mean / p90 / max | 0 / 345 / 398 / 487 ms | 0 / 372 / 431 / 491 ms | mean, p90 ≤ 500 | pass |
| Q2 simulated elderly (80): p90 / max | 393 / 465 ms | 446 / 546 ms | p90 ≤ 500 | pass |
| Q3 coherence after interruption: relevance, interrupt vs control | Gemini 2.73 vs 2.70; Qwen3.8 3.27 vs 3.07 | Gemini 1.67 vs 1.67; Qwen3.8 2.90 vs 2.80 | interrupt ≥ control − 0.25, both judges | pass |
| Q4 Latin letters in replies | 0 of 428 | — | none | pass |

Every planned turn of both panels was answered within 500 ms (audible p90,
including the browser's output latency: 440 ms plain, 435 ms elderly). The new
responder is faster to first audio although its first token comes later (median
122 vs 104 ms after the speculative start, ASR included): two thirds of its
replies open with the short address «عزیزم،» (202 of 308), so the first clause is
complete sooner (161 vs 192 ms) and its audio is ready at 222 vs 249 ms (server
medians from the run telemetry).

In the coherence scenarios every interruption landed while the reply was
playing and stopped it (30/30; v2: 24 of 27), the history kept a median of 4
heard words, and no follow-up reply was missing. Paired differences
(interrupt − control) with 95% bootstrap intervals: relevance +0.03 (−0.30…+0.37)
under Gemini and +0.20 (−0.27…+0.67) under Qwen3.8. The absolute scores of the
same scenarios are higher than the deployed responder's in v2 under both judges
(Gemini relevance 2.73 against 1.67), in line with responder comparison v2;
the two runs are separate measurements.

**Part 2 passes** (Q1–Q4). Whether the responder switches now depends only on
part 1.

## Part 1 — human preference

Pending: computed by `scripts/ingest_listening_study.py` (`responder_switch_rule`
in `results/eval/listening_study.json`) once the study codes arrive.

## Claim boundary

Automated browser loopback with recorded and synthesized speech and automatic
judges on the shared H100 host, with the memory of a 12 GiB card but not its
compute; the human part is a small remote study of written reply pairs.
