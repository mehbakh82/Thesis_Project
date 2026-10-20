# Responder switch rule — protocol

Status: frozen on 2026-10-20, before the listening-study page was sent to any
participant and before any run of the qualification below.

## Why

Responder comparison v2 (`docs/RESPONDER_CANDIDATE_COMPARISON_V2_RESULT.md`)
left a split verdict: Qwen3.5-4B with the elderly prompt never lost to the
deployed Qwen3-4B prompt-v2 under either judge and gained strongly under the
cross-family judge (Gemini relevance +0.98, elderly appropriateness 3.55 vs
2.02), but missed the pre-registered promotion bar under the Qwen3.8 judge
(+0.02). The listening study already contains the designed human tie-breaker
(eight reply pairs, `docs/LISTENING_STUDY_PROTOCOL.md`, section 2), but its
protocol said that section does not by itself change the deployed responder.
This protocol makes the decision explicit before any human answer exists.

## Decision

The deployed responder is switched to **Qwen3.5-4B with elderly prompt v1**
(prompt SHA-256 `ecaa1293680ecd0534b69b766001ea2dfd0cd3fd78b212f0e3722d0474d909b5`)
if and only if **both** parts below pass. Otherwise Qwen3-4B prompt-v2 stays
deployed and both results are reported as measured.

### Part 1 — human preference (from the listening study)

Computed by `scripts/ingest_listening_study.py` from the accepted participants
(exclusions as in the study protocol). For participant *i*, over the eight
reply pairs, `d_i` = share of pairs where the Qwen3.5-4B reply was picked minus
share where the deployed reply was picked ("no difference" counts in the
denominator for neither). Part 1 passes if all of:

1. the study meets the definition's sample: at least 5 accepted participants,
   at least 2 aged 60+;
2. the mean of `d_i` is positive and its participant-bootstrap 95% interval
   (5,000 draws, seed 20261016, as for every other study interval) lies above 0;
3. every accepted participant aged 60+ has `d_i ≥ 0`.

### Part 2 — deployment qualification (measured before the study is ingested)

One run of each measurement, with the final endpointer configuration (after
the semantic end-of-turn experiment, pre-registered separately, is decided) and
the same panels, harness, host and scorer as live loopback v2
(`docs/LIVE_LOOPBACK_V2_PROTOCOL.md`):

- **Serving:** vLLM `v0.30.0` image, Qwen3.5-4B bf16, `--gpu-memory-utilization
  0.11`, `--max-model-len 2048`, `--max-num-seqs 2`, prefix caching, image and
  video inputs disabled, thinking disabled through the chat template, the
  deployed Persian-only structured output, history of two turns, at most 96 new
  tokens (the setting under which the study's replies were generated).
- **Q1 memory:** peak GPU memory of the whole stack (LLM process + ASR
  service) at most 12,288 MiB.
- **Q2 latency:** plain panel (201 turns): no missing reply, mean and p90 of
  end of speech → first reply audio at most 500 ms (the proposal's criterion);
  simulated-elderly panel (80 turns): p90 at most 500 ms.
- **Q3 coherence after interruption:** the 30 scenarios of the v2 coherence
  measurement, interrupt and control, scored by the same two judges; met if
  the post-interruption mean relevance is at most 0.25 below the control under
  both judges (the v2 criterion).
- **Q4 script:** no Latin letter in any reply of Q2 and Q3.

Part 2 passes if Q1–Q4 all pass. It is run and recorded before the study codes
are ingested, so its outcome cannot depend on the human answers.

## If the switch happens

The live server's default responder changes to the qualified configuration,
and the qualification receipts become the deployed system's latency and
coherence figures; the Qwen3-4B runs (v2 and the end-of-turn test) stay
reported. The final semantic test (relevance 3.150 vs 1.425) was measured on
Qwen3-4B and is reported as such; the evidence for Qwen3.5-4B is responder
comparison v2, this qualification and the human preference.

## Claim boundary

Eight text pairs judged by a small remote sample; a preference for written
replies, not for spoken interaction. Qualification on the shared H100 host
with the memory of a 12 GiB card but not its compute.
