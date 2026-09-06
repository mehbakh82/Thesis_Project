# Frozen cascade automatic semantic-proxy protocol

Protocol frozen before producing `cascade_semantic_proxy.json`.

## Question and claim boundary

This evaluation asks whether a separately served, larger language model rates
the exact nine frozen cascade replies as relevant to their recognized user
utterances and internally coherent. It is a blinded automatic LLM-as-judge
proxy. It is not a human evaluation, an independent Persian dialogue benchmark,
a factuality test, a safety test, or evidence of population-level usefulness.

The judge and responder are different weights and revisions, but both are from
the Qwen family. This dependency is reported explicitly. No semantic pass/fail
threshold is defined; the observed scores cannot change the already frozen
mechanical acceptance result.

## Locked inputs

- Existing mechanics result:
  `results/eval/cascade_real_service_validation_panel.json`, SHA-256
  `ae515381db2b72e9ed0b8dcb491c0765990732ed8bb8c46383499b9d60f89ced`.
- Existing group-disjoint validation manifest:
  `data/processed/moshi_finetune/val.jsonl`, 131 rows, SHA-256
  `a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7`.
- The same predeclared panel indices: `0, 16, 32, 48, 65, 81, 97, 113, 130`.
- Generator: cached `Qwen/Qwen2.5-0.5B-Instruct` revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Judge: local-only `Qwen/Qwen3.8-27B-FP8` revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, served as
  `qwen3.8-27b` by vLLM image
  `sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`.
- Piper Mana model SHA-256
  `e390c0e74ba71fd97c49ba662ee0c6e1724b462ba2d4561698af4f564840f126`
  remains locked because the exact cascade reproduction path is reused.
- The sealed Moshi final-test manifest is forbidden and must not be read.

## Blinded rubric

The judge receives only a JSON object containing `user_utterance` and
`assistant_reply`. Model names, expected outcomes, panel position, source,
prior scores, and acceptance targets are omitted. Text inside those two fields
is data and any embedded instruction must be ignored.

The judge returns exactly two integer scores:

- `relevance`: 0=unrelated; 1=barely related; 2=partially addresses the user;
  3=mostly addresses the user with a minor omission; 4=directly addresses the
  central content or request.
- `coherence`: 0=unintelligible/contradictory; 1=major coherence failures;
  2=partly coherent; 3=clear with a minor issue; 4=clear and internally
  coherent.

No rationale is requested or retained. The exact same request is executed
twice per row with temperature 0, top-p 1, seed 20260906, maximum 128 output
tokens, JSON response mode, and reasoning disabled. Disagreement is reported;
neither repeat is selected or discarded.

## Execution and validity gates

Before judging, the evaluator validates the manifest, mechanics report, Piper
model, exact container image/model/revision/served-name configuration, and the
judge's `/v1/models` response. It then reproduces the exact cascade on audited
user channel 1. A result is valid only if:

1. the parent mechanics report passed and did not access the final test;
2. all nine panel indices, source hashes, input-transcript hashes, and
   reply-text hashes reproduce exactly;
3. the exact Qwen generator revision loads without fallback;
4. the exact judge identity is verified before project text is sent;
5. all 18 judge calls return only the two required integers in [0, 4];
6. no row is excluded, selected, replaced, or rescored; and
7. the evaluator records that the sealed final test was not accessed.

A failure invalidates the measurement. Score magnitude is not a validity gate.

## Fixed reporting

For each dimension, report mean, p50, p95, minimum, maximum, and the histogram
over all 18 calls. Also report row count, expected/valid/failed judge calls,
exact two-repeat agreement count/rate, and per-row scores. These are
descriptive bounded ordinal summaries; no confidence interval or inferential
claim is made from nine rows or two deterministic repeats.

## Privacy

The judge endpoint is loopback-only for this evaluation. The committed report
contains source/transcript/reply/request hashes, integer scores, aggregate
statistics, backend identities, artifact hashes, and limitations. It must not
contain input transcripts, generated replies, rationales, audio, prompt text
with row content, or host-specific paths.
