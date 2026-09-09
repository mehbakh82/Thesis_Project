# Frozen post-release ASR candidate comparison v1

**Frozen:** 2026-09-08; denominator erratum frozen before any selection  
**Scope:** leakage-conscious development selection followed by a separately
locked final panel  
**Privacy:** committed receipts contain hashes, aggregate error counts, timing,
and language labels only; no reference or hypothesis text

## Question

On natural Persian user turns from Digiato and Zoomit, does Qwen3-ASR-1.7B or
Qwen3-ASR-0.6B materially outperform the deployed fine-tuned NeMo Persian
service?

The incumbent's provenance document states that its then-unused YouTube pool
included Digiato and Zoomit, while about 250 hours of Tabaghe16 had already
been used for fine-tuning. Therefore this experiment excludes Tabaghe16 and
Mehran Rowshan Persian. The private external provenance document's location is
supplied explicitly at runtime and is not recorded in the repository. Its expected SHA-256 is
`307c90f98dc72e4f0ac56a5c9cdf1f5963bc06a2d4c3446c09d00ee4436d7deb`.
This removes known project-side overlap; undocumented upstream pretraining
contamination cannot be ruled out.

## Frozen inputs

- Source: `data/processed/manifests/conversations.jsonl`
- Expected SHA-256:
  `aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86`
- Sources: Digiato and Zoomit only.
- Development: four source sessions per channel, five rows per session = 40
  rows / eight sessions.
- Final: the next four independently hashed sessions per channel, five rows
  per session = 40 rows / eight different sessions.
- Selection seed: `20260908-asr-candidates-v1`.
- Eligible rows are in the conversation `train` partition, have an existing
  user-channel WAV, duration 1--20 seconds, reference length 2--80 words, and
  at least 80% Persian-script letters.
- Reference text is the automatically aligned public-video caption. It is not
  human-corrected ground truth; this limitation applies identically to all
  arms and caps the claim.

The plan stage records panel, evaluator, protocol, source, model-file, and
incumbent-container identities before transcription. Development may select a
challenger. Final remains locked unless all identities still match and a
challenger clears every frozen promotion gate.

## Frozen arms

| Arm | Runtime | Role |
|---|---|---|
| Fine-tuned Persian NeMo | deployed HTTP service at `127.0.0.1:8090` | incumbent |
| Qwen3-ASR-1.7B | official `qwen-asr` Transformers backend, BF16 | primary challenger |
| Qwen3-ASR-0.6B | official `qwen-asr` Transformers backend, BF16 | low-memory challenger |

The expected running NeMo API image ID is
`sha256:1c681f0d7cc6f192aa753efd98cedb0ef2db2c47fb8159d52912f9c03d7ad1b7`;
the Triton image ID is
`sha256:96b53a69eeff0b9783baf36eaa07d54701c7bbe652fcbe6af140f8d5569d5606`.
Qwen uses forced language `Persian`, greedy package defaults, batch size one,
and at most 256 new tokens. All text is normalized by the same project
normalizer before scoring.

## Metrics and development rule

Every row and every failure remains in the denominator. Report micro/macro
CER and WER, exact-match counts, generation latency, real-time factor, failures,
model peak CUDA allocation, and per-row hashes/error counts. Uncertainty is a
deterministic 10,000-draw paired bootstrap over source sessions.

Attempt 1 stopped fail-closed without selecting a model because the NeMo
service returned an empty string on one otherwise valid 2.19-second row and
the evaluator incorrectly omitted that row from aggregate CER/WER. The
original plan and receipt are preserved with `_attempt1` filenames. This
erratum changes no panel, candidate, output, threshold, or ranking rule: an
empty hypothesis is now scored as a full deletion, while a runtime exception
also receives full-deletion error counts and makes that arm mechanically
invalid. Thus every reference remains in every aggregate denominator.

Rank mechanically valid arms by lower micro CER, lower micro WER, lower median
RTF, then stable arm name. A challenger opens final only when all are true:

- all 40 rows complete without a runtime exception (empty hypotheses remain
  valid scored outputs and count as full deletions);
- absolute micro-CER reduction versus NeMo is at least 0.02;
- micro WER is no worse than NeMo;
- the upper endpoint of the 95% paired session-bootstrap interval for
  `challenger CER - NeMo CER` is below zero.

Otherwise retain NeMo and leave final sealed. On final, production promotion
requires complete validity, lower micro CER, non-worse micro WER, and the same
bootstrap upper-bound condition. Development selects; final confirms once.

Offline RTF is comparable engineering evidence, not streaming first-partial
latency. A future deployment promotion still needs a separately measured
streaming service/browser test.

## Commands

```bash
.venv-qwen-asr/bin/python scripts/evaluate_asr_candidates_v1.py \
  --stage plan --nemo-provenance /path/to/private-provenance.md
.venv-qwen-asr/bin/python scripts/evaluate_asr_candidates_v1.py \
  --stage development --nemo-provenance /path/to/private-provenance.md
```

Run `--stage final` only if development reports that the final lock is open.

The portable path-interface refactor occurred after this experiment was frozen.
The exact evaluator and protocol blobs bound by the receipts remain available
at the commit and hashes enforced by `configs/frozen_evaluator_sources.json`;
`scripts/verify_frozen_evaluator_sources.py` verifies both those historical
blobs and the current portable files.
