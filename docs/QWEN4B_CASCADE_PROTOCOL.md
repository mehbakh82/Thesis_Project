# Frozen Qwen3-4B cascade recovery protocol

Protocol frozen before producing `qwen4b_responder_semantic_proxy.json`.

## Purpose and claim boundary

The compact Qwen2.5-0.5B responder passed mechanical cascade checks but failed
the automatic semantic proxy. A subsequently frozen LoRA experiment reduced
development loss by 31.18% yet made held-out relevance worse; its automatically
aligned reference turns also averaged only 1.55/4 relevance. Those results are
retained. This protocol evaluates an architecture correction: replace only the
text responder with the already-local, instruction-tuned Qwen3-4B-Instruct-2507
while retaining NeMo ASR and Piper TTS.

This is an automatic engineering proxy, not human evaluation, an independent
Persian benchmark, factuality/safety evidence, population generalization, or a
physical RTX 4090/browser latency result. The judge and candidate responder are
different weights but both are Qwen-family models, an explicit dependency.

## Locked candidate

- Upstream identity: `Qwen/Qwen3-4B-Instruct-2507`, revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0.
- Exact local tree: 28 files, 8,060,919,167 bytes, SHA-256 tree digest
  `cde447f1326f10c4126061914c57c3664551649286ad6411bffe3d1aa3e3b978`.
- BF16 weights; greedy decoding; at most 64 new tokens per attempt; no fallback.
- Frozen system prompt: act as a Persian voice-dialogue assistant; answer the
  user's central question or intent directly, relevantly, naturally, and in at
  most two short sentences; use Persian script only, with no Latin letters,
  lists, or extra explanation.
- If and only if the first output fails the 0.8 Persian-letter-fraction gate,
  retry once with the frozen stricter prompt: directly answer the central intent
  in one or two short Persian sentences, with no Latin letters, code, list, or
  aside. A second failure invalidates the run; no rule response is substituted.
- A pre-protocol synthetic-only smoke test used three authored prompts, not
  project data. All three generations had Persian-letter fraction 1.0; measured
  CUDA allocation was 7,758.4 MiB. No semantic judge or validation row was used.
- The weight footprint is compatible in principle with a 24 GB RTX 4090, but
  full physical coexistence and latency remain unmeasured and unclaimed.

## Locked unseen validation panel

- Source manifest SHA-256:
  `aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86`.
- Exported validation manifest: 131 rows, SHA-256
  `a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7`.
- Selection uses the same three validation sessions reserved from responder
  development. Rows already used by either prior semantic proxy are excluded.
  Remaining rows retain the prior SHA-256 rank; the next 40 are:
  `17, 18, 19, 22, 25, 26, 27, 35, 37, 43, 44, 45, 50, 53, 56, 57, 82, 83,
  84, 88, 92, 95, 96, 98, 99, 100, 101, 103, 107, 108, 110, 112, 115, 116,
  118, 121, 125, 126, 128, 129`.
- The seven residual eligible rows remain unused.
- The 204-row test split is not used for model selection or evaluation. As
  already disclosed, a pre-protocol aggregate audit parsed combined-manifest
  test metadata/text lengths without displaying or manually inspecting text;
  the claim is therefore "test not used," not "test bytes never read."

Each locked user-channel-1 waveform is recognized by the same NeMo service.
Three replies are compared per row: unchanged Qwen2.5-0.5B, Qwen3-4B, and the
automatically aligned next-speaker reference (context only, not gold). Qwen3-4B
alone is synthesized through the exact Piper Mana model, SHA-256
`e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126`.

## Locked blinded judge and gate

The exact local `Qwen/Qwen3.8-27B-FP8` revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, served as `qwen3.8-27b` from
image `sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`,
scores relevance/coherence from 0 to 4. It receives no arm name, model identity,
row index, expected result, prior result, or threshold. Arm order is a
deterministic hash permutation. Each arm is scored twice at temperature 0,
top-p 1, seed 20260906, JSON-only output, and reasoning disabled. No rationale
is requested or retained.

Measurement validity requires exact data/model/judge/Piper hashes, exact panel,
40 successful ASR calls, base and candidate generation without fallback,
Persian-script gates, valid Piper audio, all 240 judge calls, and test not used.
Score magnitude cannot change measurement validity.

The automatic engineering success gate intentionally matches the preceding
LoRA protocol:

- valid evidence;
- Qwen3-4B relevance mean at least 2.0/4;
- Qwen3-4B coherence mean at least 2.5/4;
- Qwen3-4B minus base relevance gain at least 0.50;
- Qwen3-4B minus base coherence gain non-negative;
- Qwen3-4B relevance median exceeds base on at least 60% of rows; and
- Qwen3-4B relevance median is at least 2 on at least 70% of rows.

The committed output contains only hashes, ordinal scores, aggregate metrics,
backend/artifact identities, and limitations. It contains no plaintext input,
reply, reference, rationale, audio, or host-specific model path. All model
traffic remains loopback-only.
