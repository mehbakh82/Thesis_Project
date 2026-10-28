# Responder size comparison — protocol

Status: frozen on 2026-10-28, before any reply of a new arm was generated.

## Why

Every responder compared so far had at most 4 billion parameters, because the
closeout kept the whole stack inside a 12 GiB card. The proposal's hardware
range goes up to 24 GB, and at 4 bits much larger models fit that range. This
comparison asks whether a larger responder, quantized where needed to fit a
24 GB card, gives better replies than the best 4B responder (research question
3: the trade-off between size, latency and quality).

## Panel and judges (unchanged from responder comparison v2)

The 120-row v2 panel (60 hand-written elderly-assistant requests, 60 natural
user turns; `docs/RESPONDER_CANDIDATE_COMPARISON_V2_PROTOCOL.md`), the v2
rubric verbatim (relevance, coherence and elderly appropriateness, 0–4), and
both v2 judges: the local Qwen3.8-27B and Gemini 3.8 Flash, blind, in random
order. The panel was already used by v2; no prompt is tuned here. Every arm
except the incumbent uses the frozen elderly-assistant prompt v1.

## Arms

| Arm | Checkpoint (pinned) | Weights |
|---|---|---|
| `qwen3_4b_prompt_v2` (deployed) | Qwen3-4B-Instruct-2507, deployed prompt v2 | bf16 |
| `qwen35_4b_elderly_v1` (best 4B, reference) | Qwen3.5-4B @ `851bf6e8…` | bf16 |
| `gemma4_e4b_elderly_v1` | Gemma 4 E4B, official QAT | 4-bit (w4a16) |
| `qwen35_9b_elderly_v1` | `Qwen/Qwen3.5-9B` @ `c2022362…` | bf16 |
| `gemma4_12b_elderly_v1` | `google/gemma-4-12B-it-qat-w4a16-ct` | official 4-bit QAT |
| `gemma4_26b_a4b_elderly_v1` | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` @ `76373001…` (mixture of experts, 4B active) | 4-bit AWQ of Google's QAT weights (no official vLLM-ready 4-bit release exists) |
| `qwen35_35b_a3b_elderly_v1` | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` @ `3af5ca29…` (mixture of experts, 3B active) | official 4-bit GPTQ |
| `gemma4_31b_elderly_v1` | `google/gemma-4-31B-it-qat-w4a16-ct` @ `52f3f65b…` | official 4-bit QAT |

Not included: Qwen3.5-27B (its official 4-bit release holds 28.2 GiB of
weights, more than a 24 GB card); Qwen3.8-27B (it is the judge, and its FP8
weights exceed 24 GB); Qwen3.8-Flash-Next (180B parameters); NVFP4 releases
(need Blackwell GPUs).

## Generation (the deployed serving path, the same for every arm)

The three 4B arms are generated again through this path, so every arm is
compared under identical serving. vLLM `v0.30.0` image; greedy decoding, at most
96 new tokens, no history; the deployed Persian-only structured output
(`PERSIAN_ONLY_REGEX`); thinking disabled through the chat template where the
model supports it; image, video and audio inputs disabled; `--max-model-len
2048`, `--max-num-seqs 2`. Each reply is streamed, and the time to the first
token and to the last token is recorded.

**Fit to a 24 GB card:** each arm runs with `--gpu-memory-utilization` set to
22 GiB of the H100's memory, leaving room for the ASR service and the CUDA
contexts of a 24 GB card. An arm that cannot start or serve under this limit is
recorded as *does not fit* and is not judged. The peak memory of the vLLM
process is recorded.

## Decision rules

Paired comparisons use the v2 statistics: mean relevance gain with a 5,000-draw
row bootstrap interval, coherence and appropriateness gains, and the row win
rate.

1. **Does size buy quality?** A larger arm *beats the best 4B* if, under
   **both** judges, against `qwen35_4b_elderly_v1` its relevance gain is at
   least +0.25, its coherence and appropriateness gains are non-negative, and
   it wins at least 55% of rows (the v2 promotion rule with the best 4B as
   the reference). The same rule against the deployed responder is also
   reported.
2. **Deployment.** The listening study still decides between the two 4B
   responders (`docs/RESPONDER_SWITCH_PROTOCOL.md`); the human evidence covers
   only those two. If one or more larger arms beat the best 4B, the arm with
   the highest relevance gain averaged over the two judges is qualified as the
   responder of a **24 GB profile**, with the switch protocol's part 2
   measurements: Q1 with a 22 GiB limit for the LLM process plus the ASR
   service, and Q2–Q4 unchanged (loopback latency, coherence after
   interruption, no Latin letters). If it passes, `serve-live` gains a 24 GB
   preset with that responder, and the 12 GiB default stays as the study
   decides. If no larger arm beats the best 4B, the result is reported as
   evidence that, on this panel, size did not buy reply quality.

## Claim boundary

Automatic judges only (one same-family judge for the Qwen arms and one for the
Gemma arms, which is why both must agree); 120 written requests and turns, not
spoken interaction; serving speed on the shared H100, not on a 24 GB card.
