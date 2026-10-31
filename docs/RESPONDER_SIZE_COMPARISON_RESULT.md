# Responder size comparison — result

Protocol: `docs/RESPONDER_SIZE_COMPARISON_PROTOCOL.md`. Receipt:
`results/eval/responder_size_comparison.json`. **Outcome: no larger responder
beats the best 4B under both judges, so size did not buy reply quality on this
panel and no 24 GB profile is qualified. Gemma 4 31B at 4 bits does not fit
the 22 GiB limit.**

## Results

The v2 panel has 120 rows (60 elderly-assistant requests and 60 natural user
turns). All eight arms were served by vLLM v0.30.0 under a 22 GiB limit, with
greedy decoding, the Persian-only structured output and at most 96 new tokens.
Both judges scored every reply blind, on a 0–4 scale. Gains are paired
differences against the best 4B (`qwen35_4b_elderly_v1`), with a 5,000-draw
row bootstrap interval. A row is a win when the larger arm scores higher on
relevance.

| Arm | Params | Weights | Qwen3.8 relevance (gain, 95% CI) | Gemini relevance (gain, 95% CI, win) | First token | Full reply |
|---|---:|---|---|---|---:|---:|
| Qwen3-4B, prompt v2 (deployed) | 4.02B | bf16 | 3.25 | 2.46 | 38 ms | 170 ms |
| **Qwen3.5-4B (best 4B, reference)** | 4.66B | bf16 | 3.23 | 3.35 | 68 ms | 267 ms |
| Gemma 4 E4B | 8.61B¹ | 4-bit QAT | 2.88 | 3.50 | 54 ms | 240 ms |
| Qwen3.5-9B | 9.65B | bf16 | 3.24 (+0.01, −0.15…0.18) | 3.48 (+0.13, −0.03…0.31, 23%) | 66 ms | 609 ms |
| Gemma 4 12B | 12.0B | 4-bit QAT | 3.04 (−0.19, −0.39…0.02) | 3.61 (+0.26, 0.08…0.44, 26%) | 56 ms | 449 ms |
| Gemma 4 26B-A4B (MoE, 4B active) | 25.8B | 4-bit AWQ | 2.77 (−0.46, −0.66…−0.26) | 3.55 (+0.20, 0.03…0.38, 24%) | 53 ms | 201 ms |
| Gemma 4 31B | 31.3B | 4-bit QAT | does not fit | does not fit | — | — |
| Qwen3.5-35B-A3B (MoE, 3B active) | 36.0B | 4-bit GPTQ | 3.13 (−0.10, −0.28…0.08) | 3.49 (+0.14, −0.03…0.31, 23%) | 121 ms | 292 ms |

¹ Stored parameter count, including the per-layer embeddings; the name refers
to about 4B effective parameters.

Exact agreement between the judges on relevance is 52.4%. No reply was empty,
and none contained a Latin letter.

## Decision (pre-registered rule)

To beat the best 4B, an arm needed all four of the following under **both**
judges: a relevance gain of at least +0.25, non-negative coherence and
appropriateness gains, and wins on at least 55% of rows.

- **Qwen judge:** no larger arm gains more than +0.01 in relevance. Gemma 4
  26B is worse, and its interval excludes zero.
- **Gemini:** every larger arm gains. Gemma 4 12B reaches the threshold
  (+0.26), but it wins only 26% of rows, because most rows tie.
- **Result:** `size_buys_quality` is false and `profile_24gb_candidate` is
  null. None of the larger arms beats the deployed responder under both judges
  either. The listening study still decides between the two 4B responders, as
  `docs/RESPONDER_SWITCH_PROTOCOL.md` specifies.

## Memory fit

- **Gemma 4 31B** does not fit. Its 4-bit weights load in 18.7 GiB, which
  leaves 0.05 GiB of the 22 GiB limit for the KV cache. One request at the
  protocol's 2048-token context needs 1.73 GiB, so vLLM refuses to start. As
  the protocol specifies, it is recorded as *does not fit* and is not judged.
- **Gemma 4 12B** first failed to start because video input was left enabled,
  against the protocol's settings. It started on the second attempt with
  image, video and audio disabled, and that run is the measurement. The
  receipt notes both attempts.
- **Memory readings:** vLLM preallocates up to its limit, so the measured
  process memory (20.9–22.4 GiB) shows the limit, not what each model needs.

## Interpretation

- **The judges split along model families, as in v2.** Gemini prefers the
  Gemma arms (+0.20 and +0.26), and the Qwen judge prefers the Qwen arms. The
  protocol requires both judges to agree for this reason, so the split leaves
  no candidate.
- **Larger Gemma models score higher on coherence and appropriateness.** Both
  judges agree on this (Gemma 4 26B: +0.15 and +0.18 under the Qwen judge,
  +0.12 and +0.38 under Gemini). They do not score higher on relevance, which
  is the primary measure.
- **Latency does not rule the larger models out.** Every arm's first token
  arrives within 53–121 ms. The bf16 9B is the slowest to finish a reply
  (609 ms for at most 96 tokens). The mixture-of-experts arms finish faster
  than the dense 12B.
- **The 4B reruns are consistent with v2.** The three 4B arms were generated
  again through this serving path. Their scores are within 0.12 of the v2
  receipt (`results/eval/responder_candidates_v2_two_judges.json`).

## Claim boundary

Automatic judges only, one of them of the same family as each model group;
120 written requests and turns, not spoken interaction; serving speed on the
shared H100 under a 22 GiB limit, not on a 24 GB card.
