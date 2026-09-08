# Frozen post-release responder candidate comparison v1

**Frozen:** 2026-09-08, before candidate outputs were generated  
**Scope:** development selection followed by a separately locked final panel  
**Privacy:** committed receipts contain hashes, aggregate metrics, timings, and
scores only; no archive transcript or generated reply text

## Question

On an equal-channel Persian conversation panel that is disjoint by source
session from the project validation and test partitions, does local Qwen3.5-4B
or Qwen3.5-0.8B provide a material response-quality improvement over the frozen
Qwen3-4B-Instruct-2507 prompt-v2 responder?

This comparison does not alter or reuse the already-open 40-row final panel.
It uses only rows marked `train` in the immutable natural-source conversation
manifest. It is a post-release component experiment, not a replacement for the
submitted thesis result.

## Frozen inputs

- Source: `data/processed/manifests/conversations.jsonl`
- Expected SHA-256:
  `aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86`
- Channels: Digiato, Mehran Rowshan Persian, Tabaghe16, Zoomit
- Development: two source sessions per channel, five rows per session = 40
  rows / eight sessions.
- Final: the next two independently hashed sessions per channel, five rows per
  session = 40 rows / eight different sessions.
- Within each channel and session, selection is SHA-256 ordered with seed
  `20260908-responder-candidates-v1`.
- Eligible rows require nonempty user and reference response text, 2--80 user
  words, 2--80 response words, and at least 80% Persian-script letters in both.

The `plan` stage records both panel hashes before any inference. Development
may select a model. Final remains locked until the development receipt and all
protocol/evaluator/source hashes match. If development retains Qwen3-4B, the
new final panel remains unopened because the existing frozen Qwen3-4B result
already establishes the production decision.

## Frozen candidates

| Arm | Revision | Role |
|---|---|---|
| Qwen3-4B-Instruct-2507 prompt-v2 | `cdbee75f17c01a7cc42f958dc650907174af0554` | incumbent |
| Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | upgrade candidate |
| Qwen3.5-0.8B | `23c69c53358a07516b5827588b3fdb12ae78fd65` | low-memory candidate |

All arms use the exact prompt-v2 instruction already frozen for the incumbent,
greedy decoding, thinking disabled where supported, and at most 64 new tokens.
No candidate-specific prompt tuning is permitted.

## Judge and metrics

- Judge: exact local Qwen3.8-27B-FP8 service identity `qwen3.8-27b`.
- Temperature: zero; thinking disabled; two calls per row/arm.
- Dimensions: relevance and coherence on the existing 0--4 rubric.
- Mechanical validity: every row generates a nonempty reply with Persian-letter
  fraction at least 0.80 and receives both valid judge calls.
- Report per arm: means, histograms, repeat agreement, generation p50/p95/max,
  peak process CUDA allocation, failures, and response hashes.
- Report pairwise against incumbent: mean relevance/coherence gains and the
  row-level median relevance win rate.

The automatic same-family judge is a proxy. It is not human naturalness,
factuality, safety, or population evidence.

## Frozen development selection rule

Among mechanically valid arms, rank by:

1. higher mean relevance;
2. higher mean coherence;
3. lower median generation time;
4. stable arm name as the final deterministic tie-break.

A challenger replaces the incumbent for the locked final stage only when all
of these are true:

- relevance gain over incumbent is at least 0.25;
- coherence gain is nonnegative;
- row-level median relevance win rate is at least 0.55;
- every mechanical validity requirement passes.

Otherwise retain Qwen3-4B and do not open the new final panel.

## Commands

```bash
.venv/bin/python scripts/evaluate_responder_candidates_v1.py --stage plan
.venv/bin/python scripts/evaluate_responder_candidates_v1.py --stage development
```

Run `--stage final` only if the development receipt promotes a challenger and
the script reports that the final lock is open.

