# Responder comparison v3: models, quantized variants and serving settings (protocol)

Status: frozen on 2026-10-31, before any reply of this comparison was generated.

## Why

The responder comparisons so far left four gaps. Both judges came from candidate
families (Qwen judging Qwen, Gemini judging Gemma), and they agreed on only 44.6%
of relevance scores. The panel was small: 120 rows, with the 60 elderly requests
written by one author, the AI assistant that ran the closeout. Only a few
quantized builds were tried, all official except one. And the serving settings
were fixed, so "does not fit" for Gemma 4 31B held only under those settings.
This comparison covers Qwen3.5, Qwen3.6, Qwen3.8 and Gemma 4 with every
quantization method that the two serving engines can run, official and
community, tunes the serving settings of each, and judges with two models from
families that have no candidate.

## Question

Within each deployment profile, which responder configuration (model,
quantized weights, engine and serving settings) gives the best replies? Does it
beat the best 4B responder (Qwen3.5-4B, bf16, elderly prompt) and the deployed
one (Qwen3-4B-Instruct-2507, prompt v2)?

- **12 GiB profile:** the whole stack in 12,288 MiB. The speech recognizer
  takes 1,606 MiB (`docs/LIVE_LATENCY_LOOPBACK_RESULT.md`), so the responder's
  footprint must be at most 10,682 MiB.
- **24 GB profile:** the responder's footprint must be at most 22,528 MiB
  (22 GiB), which leaves room for the recognizer and the CUDA contexts on a
  24,576 MiB card.

Deployment requirement for every configuration: a context of at least 2,048
tokens, two concurrent sequences (the speculative start can overlap the final
request), and 96 new tokens.

## Models

The official instruct checkpoints that fit a 24 GB card at some precision:

| Family | Checkpoints |
|---|---|
| Qwen3.5 | 0.8B, 2B, 4B, 9B, 27B, 35B-A3B |
| Qwen3.6 | 27B, 35B-A3B |
| Qwen3.8 | 27B |
| Gemma 4 | E2B, E4B, 12B, 26B-A4B, 31B (instruction-tuned `-it`) |

Out of scope, with the reason recorded in the inventory: Qwen3.5-122B-A10B and
397B-A17B, Qwen3.8-Flash-Next and Qwen3.8-2.4T (too large even at 3 bits); base
(pretrained) checkpoints; Gemma 4 `-assistant` checkpoints, which are drafters
for speculative decoding and are used only as a serving option. Qwen3-4B
(deployed) and Qwen3.5-4B bf16 with the elderly prompt are the two references.

## Variants (selection rule)

`scripts/inventory_responder_variants_v3.py` writes
`results/eval/responder_v3_inventory.json` from the Hugging Face Hub on the date
above. Download counts change daily, so this inventory is the frozen arm list.

1. **Official releases:** bf16, FP8, GPTQ-Int4, Google's QAT w4a16 and QAT
   q4_0 GGUF.
2. **Community builds:** repositories that declare themselves a quantization
   of the official instruct checkpoint (`base_model:quantized`), or of
   Google's QAT weights. A repository is eligible if its name carries the model
   name, it is not a fine-tune, merge, behaviour edit (abliterated, uncensored)
   or pruned model, and it is packaged for vLLM or llama.cpp. Per method family
   (FP8, INT8 W8A8/W8A16, AWQ INT4, GPTQ INT4, AutoRound INT4,
   compressed-tensors W4A16, NVFP4, MXFP4) the selection is the most-downloaded
   repository whose own quantization config matches the family and whose files
   total at most 26 GiB. The files include vision and audio towers that
   text-only serving skips. Up to two less-downloaded repositories of the same
   family are kept as fallbacks, tried in order if the selected one fails to
   load. An official release replaces the community pick of its family.
3. **GGUF ladder:** Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q3_K_M and Q2_K, plus
   the dynamic UD-Q4_K_XL and UD-Q2_K_XL. Each rung comes from the
   most-downloaded general publisher that has it (unsloth, bartowski,
   lmstudio-community, ggml-org, mradermacher, QuantFactory; repositories
   with built-in speculative heads or QAT weights are not used for the ladder).
   Each rung must total at most 21.5 GiB.
4. **Ternary and binary builds** (PrismML Bonsai, quantization-aware
   trained): these need the publisher's llama.cpp fork. They are tested only
   if the fork can be built on this host; otherwise they are recorded as not
   tested.

Not tested, for reasons verified on this host: bitsandbytes (vLLM `v0.30.0`
has no bitsandbytes method); GGUF in vLLM (removed in `v0.30.0`; llama.cpp
serves GGUF instead); MLX, ONNX, EXL2/EXL3 and NPU packages (other runtimes);
builds larger than the limits above.

## Engines and generation (the same for every arm)

- vLLM `v0.30.0` for safetensors builds; llama.cpp build 11446 (commit
  `a043d38a6`) for GGUF. Neither ever runs remote code.
- The frozen elderly-assistant prompt v1 for every arm except the deployed
  reference, which keeps prompt v2; a two-turn history where the panel row has
  one; greedy decoding; at most 96 new tokens; thinking disabled through the
  chat template; text-only serving (vLLM `--language-model-only`, which does not
  load the vision and audio encoders; llama.cpp without a projector file).
- The deployed Persian-only constraint: in vLLM the regular expression
  `PERSIAN_ONLY_REGEX` as structured output; in llama.cpp an equivalent GBNF
  grammar over the same character class. Replies containing Latin letters are
  counted.

## Serving settings (every arm)

- **vLLM:** KV cache precision {auto (bf16), fp8}; context {2,048, 4,096,
  8,192}; concurrent sequences {1, 2, 4}; 2,048 batched tokens; prefix caching
  on; CUDA graphs on. Eager mode is excluded because it is markedly slower.
- **llama.cpp:** all layers on the GPU; flash attention; KV cache type {f16,
  q8_0, q4_0}; context {2,048, 4,096, 8,192} × slots {1, 2, 4}.
- **Footprint:** the KV memory one sequence needs depends on the model's
  architecture and the KV precision, not on how its weights are quantized. It
  is measured once per model and engine: one start at 2,048 and one at 8,192
  tokens for each KV precision, giving a constant (recurrent state, sliding
  windows) and a per-token term. Each arm is then started at the deployment
  configuration with its KV cache sized exactly for that requirement (vLLM
  `--kv-cache-memory-bytes`; llama.cpp context × slots). The process's peak
  GPU memory, read with `nvidia-smi` while it serves, is its footprint, and
  the footprint minus the KV allocation is its fixed part. The footprint of
  any context, concurrency and KV precision is the fixed part plus
  concurrency × the per-sequence KV. This formula is checked by starting one
  arm per architecture (Qwen dense, Qwen mixture-of-experts, Gemma dense,
  Gemma mixture-of-experts, Gemma E-series) at the grid corners (2,048 and
  8,192 tokens × 1 and 4 sequences × every KV precision). The finalists'
  configurations are confirmed by a start at exactly that configuration. A
  configuration fits a profile if its footprint is within the profile's
  limit.
- **Latency** at the deployment configuration: 40 rows of the development
  panel one at a time (time to the first token, to the first clause, to the
  last token; decode rate), and 20 bursts of two concurrent requests.
- **KV precision and quality:** development replies are generated with the
  default KV precision for every arm, and with every other KV precision for
  each model's best variant and for any finalist whose configuration needs a
  lower KV precision to fit.
- **Speculative decoding** (Qwen's MTP head, Gemma's assistant drafter):
  measured for the finalists only, as a latency option. With greedy decoding
  it does not change the reply.
- **Reference precision:** where the full or FP8 weights can run on the shared
  H100 (the FP8 Qwen3.8-27B already served there; FP8 Qwen3.5-27B, Qwen3.6-27B
  and Gemma 4 26B-A4B and 31B when the paused services free enough memory),
  the same generation runs at that precision. These arms do not fit either
  profile; they only measure what quantization costs.

## Order of work and what the decision uses

Downloads from the Hugging Face Hub run at about 1.7 MiB/s on this host
(measured on 2026-10-31 and in earlier fetches), so the 245 builds (about
2.3 TiB) would take about 16 days. The arms run in this order, each tier
finishing before the next starts (`scripts/evaluate_responder_v3.py --stage
arms` assigns the tiers):

- **A1:** the two references and the builds already on the host;
- **A2:** Q4_K_M of every model, the same GGUF rung across models;
- **A3:** for every model and profile, the highest-precision build that fits in
  each engine, and Google's QAT GGUF;
- **B:** every other build of Qwen3.5-4B, Qwen3.5-9B and Gemma 4 12B, which is
  every method and the whole GGUF ladder on three models;
- **D:** every other build.

Quantization only removes information, so for each model and profile the
highest-precision build that fits is expected to give the best replies. Tier A
therefore holds each model's expected best configuration. Tier B measures how
much the method and the bit width matter, and so whether a lower-precision
build from tier D could change a decision. The variant screen and the
finalists (stages 4 and 5) run when tiers A and B are complete, over every arm
finished by then. Tier D keeps running; each arm finished later is screened the
same way, and if it would have been a finalist, it is added to the test stage
and the result says so.

## Amendment 1 (2026-10-31): adaptive selection replaces tiers A3, B and D

Written after the user declined a faster transfer path and asked for a sharper
filter, while only tier A1 (builds already on the host) had run, so before any
affected reply existed. Tiers A1 and A2 are unchanged and form stage 1; tiers
A3, B and D are replaced by stages 2 and 3. The panels, judges, serving grid,
finalist rule (stage 5) and decision rule (stage 7) are unchanged.

- **Stage 1, model screen:** every model at Q4_K_M (tiers A1 and A2), so the
  model is the only difference, plus Qwen3.8-27B at FP8 served by the existing
  local `qwen3.8-27b` vLLM service (requests only, nothing restarted) as a
  near-full-precision large model.
- **Stage 2, precision and method sensitivity:** every other build of
  Qwen3.5-4B (every method family and every GGUF rung); Gemma 4 E4B at bf16,
  Q8_0 and NVFP4 next to Google's QAT builds; and Qwen3.8-27B at FP8 against
  its Q4_K_M. For each, the development replies are scored (J2, all variants of
  the model in one call) and compared with the model's highest-precision
  variant. Δq is the largest overall gain of a model's highest-precision
  variant over its Q4_K_M.
- **Stage 3, contenders:** a model is a contender for a profile if it has a
  build that fits the profile and its best stage 1 score (J2, development
  panel, all stage 1 arms scored together) is within δ = max(0.5, 2·Δq) of the
  best stage 1 score among models with a fitting build. Quantization only
  removes information, so a model further behind cannot catch up with more
  precision. For each contender and profile, stage 3 tests the highest-precision
  build that fits in each engine and, if different, the build of the method
  that scored best at 4 bits in stage 2. Rungs at 3 bits or fewer are tested
  only if Q3_K_M was non-inferior (within 0.3) in stage 2.
- Every build that is not tested is listed in the result with the reason
  (model outside δ, or method or bit width dominated in stage 2). Serving
  settings are computed for every tested build; KV precision quality is
  measured on the stage 2 models and the finalists.

## Amendment 2 (2026-10-31): history turns are cut to 40 words

The first development run failed: some natural turns carry two whole caption
windows as history, up to 3,215 tokens, more than the 2,048-token deployment
context. The deployed runtime's history holds the assistant's own short
replies and short user turns. So every history turn (user and assistant) is cut
to its last 40 words when the prompt is built, for generation and for the
judges alike. The panel files and their hashes are unchanged. No reply had been
generated when this was decided.

## Panels

Frozen by `scripts/evaluate_responder_v3.py --stage plan` with SHA-256 hashes
before any generation. Natural-turn text is corpus text and stays out of the
repository; the receipts record hashes.

- **Elderly requests (176):** colloquial requests an older Persian speaker
  might say to a home voice assistant, 11 per author in each of the eight v2
  categories. 88 are written by Claude (the assistant running this
  comparison) and 88 by GPT-6 Astra, both from the same written brief
  (`configs/responder_v3_elderly_requests.json`). The author is recorded for
  every row.
- **Natural turns (176):** real user turns of 4–30 words, at least 80%
  Persian letters, with the two preceding exchanges of the same session as
  history, at most one per session, from the balanced-v2 manifest. Rows of the
  v2 panel are excluded, so no row is judged in two comparisons. None of the
  candidates was trained on this corpus, so earlier training use does not
  matter here.
- **Development:** 50 elderly requests (25 per author) and 50 natural turns.
  **Test:** 126 elderly requests (63 per author) and 126 natural turns. Rows
  are assigned by SHA-256 order with seed `20261031-responder-v3`.

## Judges

- **J1:** Claude Opus 5.5 (`anthropic/claude-opus-5.5` through the gateway,
  temperature 0). **J2:** GPT-6 Astra (Codex CLI `0.156.1` on the user's
  ChatGPT account, reasoning effort medium). Neither family has a candidate.
  Each judge's family wrote half of the elderly requests, so every result is
  also reported per author.
- **Listwise scoring:** for one row, the judge sees the history and the user's
  turn, then up to 12 anonymized replies in random order. It scores each reply
  from 0 to 10 for relevance, coherence, appropriateness for an older listener,
  and overall, and returns JSON. Identical replies are shown once, and the
  score applies to every arm that gave that reply.
- **Planted control:** every call includes one reply written for a different
  row. A judge whose control scores 6 or more overall in more than 10% of calls
  is reported as unreliable.

## Stages and decision rules

1. **Inventory** (frozen above).
2. **Fit and latency** for every arm and serving setting.
3. **Development generation** for every arm that fits the 24 GB profile in at
   least one setting, plus the reference-precision arms.
4. **Variant screen (J2, development panel):** all variants of one model in one
   call (split into calls sharing two anchors if there are more than 12). For
   each model the result is every variant's mean overall score and its paired
   difference from the model's highest-precision variant (5,000-draw row
   bootstrap). A variant is non-inferior if the difference is at least −0.3.
5. **Finalists:** for each model and profile, the fitting configuration with
   the highest development score (ties within 0.1 go to the lower median
   first-token time). These, with the two references, are scored together on
   the development panel by J2. The top six per profile go to the test panel.
6. **Test (J1 and J2, test panel):** generation and listwise scoring of the
   finalists and the two references.
7. **Decision.** The winner of a profile is the finalist with the highest
   overall score averaged over the two judges among those that fit the
   profile and meet the latency gate (median first token ≤ 150 ms and decode ≥
   30 tokens/s on the H100 at the deployment configuration). It **beats** a
   reference if, under both judges, its paired mean overall gain is at least
   0.5 points with a 95% bootstrap interval above zero, and its relevance,
   coherence and appropriateness gains are non-negative.

**Deployment.** The listening study still decides between the two 4B
responders (`docs/RESPONDER_SWITCH_PROTOCOL.md`). If the 24 GB winner beats
Qwen3.5-4B, it is qualified as the responder of a 24 GB preset with the switch
protocol's part 2 measurements (Q1 with the 22 GiB limit) and added to
`serve-live`. If the 12 GiB winner beats Qwen3.5-4B, it is reported as the
recommended 12 GiB responder for later human evaluation.

## Claim boundary

Automatic judges only; written requests and caption turns, not spoken
interaction; latency on the shared H100, not on a 24 GB card; community builds
as published on the inventory date.
