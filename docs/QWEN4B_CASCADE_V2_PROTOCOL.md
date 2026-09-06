# Frozen Qwen3-4B prompt-v2 eligibility and final-test protocol

Protocol frozen before prompt-v2 touches the seven residual validation rows and
before any final-test audio/text is evaluated.

## Motivation and immutable prior result

Qwen3-4B prompt v1 produced a valid, large relevance improvement on 40 unused
validation rows: 1.100 -> 2.425 (+1.325), 67.5% paired wins, and 70% of rows at
relevance >=2. Six of seven predeclared gates passed. The overall gate remained
negative because mean coherence was 2.400 versus the frozen 2.500 requirement.
That report is retained unchanged. Only aggregate results and retry counts were
inspected; no row text or per-row score was used to design prompt v2.

## Locked model and prompt-only correction

- Exact `Qwen/Qwen3-4B-Instruct-2507` revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0, local tree SHA-256
  `cde447f1326f10c4126061914c57c3664551649286ad6411bffe3d1aa3e3b978`
  (28 files; 8,060,919,167 bytes).
- BF16, greedy, maximum 64 new tokens per attempt, no rule fallback.
- Primary prompt v2: the input may be imperfect speech recognition; infer the
  central understandable intent and write exactly one complete, fluent Persian
  sentence of at most 25 words that directly answers it; do not repeat the
  input, leave the sentence unfinished, use Latin letters, make a list, or add
  an aside; if no intent is understandable, ask for repetition in one sentence.
- If the first output fails the 0.8 Persian-letter-fraction constraint, retry
  once with a fixed stricter one-complete-sentence Persian prompt. A second
  failure invalidates the run. No answer is substituted.
- NeMo ASR, audited user channel 1, Piper Mana, generator settings, judge,
  rubric, blinding, two repeats, and all score thresholds remain unchanged.

## Stage A: residual validation eligibility

The seven validation rows never scored by the baseline semantic panel, LoRA
panel, or Qwen3-4B v1 panel are locked as:
`23, 24, 33, 40, 47, 54, 122`.

All three arms (base 0.5B, prompt-v2 4B, automatically aligned reference) are
judged twice, for 42 required calls. Prompt v2 is eligible for the final test
only if evidence is mechanically valid and every unchanged gate passes:

- candidate relevance mean >=2.0/4;
- candidate coherence mean >=2.5/4;
- candidate-minus-base relevance gain >=0.50;
- candidate-minus-base coherence gain >=0;
- candidate median relevance beats base on >=60% of rows; and
- candidate median relevance is >=2 on >=70% of rows.

Seven rows are an engineering eligibility check, not a generalization claim.
If any gate fails, Stage B is forbidden and the experiment ends negative.

## Stage B: single-use final test

Only a passed, committed Stage-A report unlocks Stage B. No code, prompt,
threshold, model, judge, or panel change is allowed between stages.

- Test export: 204 rows, five source sessions, SHA-256
  `44d5912201ed359dabe3c026b6ae605b3bf946538e83116f57514448ed0794fe`.
- The panel was selected before Stage A using only session and utterance IDs:
  sort sessions and rows by SHA-256 under seed
  `20260906-qwen4b-v2-final-test`, then take eight rows per session.
- Locked indices:
  `1, 4, 7, 9, 12, 13, 17, 30, 31, 32, 33, 34, 35, 37, 40, 42, 74, 79, 90,
  92, 98, 106, 109, 122, 129, 133, 141, 142, 148, 160, 167, 169, 173, 183,
  185, 191, 192, 194, 200, 202`.
- The five session hashes/counts are recorded in the evaluator; the exported
  source-pair ID sequence SHA-256 is
  `b0de78cf27a112dbe3d7775940e6901e017f8d6296b9d537d75380a71fcb73dc`.

The same three arms and two repeats require 240 valid calls. The final automatic
engineering gate is identical to Stage A. Stage B is the first model-outcome
use of these test rows. A pre-protocol aggregate audit had parsed combined-
manifest test metadata and text lengths without displaying or manually
inspecting text; this limitation remains explicit.

## Judge, validity, privacy, and claim boundary

The exact local judge is `Qwen/Qwen3.8-27B-FP8` revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, served as `qwen3.8-27b` from
image `sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`.
Arm identity/order, model name, row index, expected outcome, and thresholds are
hidden from it. Traffic is loopback-only; no rationale is requested or stored.

Validity requires all exact artifact/identity/panel/stage gates, successful ASR,
no responder fallback, Persian-script replies, valid Piper audio, and every
judge call. Score magnitude affects pass/fail but not validity.

Reports store hashes, ordinal scores, aggregates, and identities only—never
plaintext transcripts/replies/references, rationales, audio, or host model
paths. Even a passed test is an automatic same-family LLM-as-judge result, not
human quality, an independent benchmark, factuality/safety, population utility,
or physical RTX 4090/browser latency evidence.
