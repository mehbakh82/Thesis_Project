# Frozen post-release Persian TTS candidate comparison v1

**Frozen:** 2026-09-08, before any project-panel synthesis

**Scope:** automatic development selection followed by a separately locked
final panel; human listening remains an explicit external limitation

**Privacy:** committed receipts contain hashes, error counts, timings, source
labels, and text lengths only; no archive text, ASR hypothesis, or audio

## Question

On identical, project-representative Persian response text, does Meta
MMS-TTS-Persian provide a material automatic intelligibility improvement over
the deployed Mana Persian Piper voice without unacceptable failures or render
latency?

This is a component comparison. NeMo round-trip CER/WER is an automatic proxy,
not a human naturalness, pronunciation, speaker-preference, or accessibility
result. The repository remains Apache-2.0, while the externally downloaded MMS
weights remain CC-BY-NC-4.0 and are never redistributed here.

## Frozen inputs

- Source: `data/processed/manifests/conversations.jsonl`
- Expected SHA-256:
  `aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86`
- Channels: Digiato, Mehran Rowshan Persian, Tabaghe16, Zoomit.
- Candidate texts are train-partition `response_text` values with 2--25 words
  and at least 80% Persian-script letters.
- Every source session used by either responder-candidate panel is excluded.
- Development: two remaining source sessions per channel, five texts per
  session = 40 texts / eight sessions.
- Final: the next two remaining source sessions per channel, five texts per
  session = 40 texts / eight different sessions.
- Sessions and rows are SHA-256 ordered with seed
  `20260908-tts-candidates-v1`.

The `plan` stage records both panel hashes, all model files, runtime versions,
the live NeMo image identities, and all evaluator dependencies before panel
inference. Development may choose whether a challenger deserves the final
panel. Final stays sealed when the incumbent wins development.

## Frozen candidates

| Arm | Exact identity | Role |
|---|---|---|
| Mana Persian Piper | ONNX/config file hashes recorded by the plan | incumbent |
| Meta MMS-TTS Persian | `facebook/mms-tts-fas` revision `8818d36618d125a0b40b5d2b2713a852877e9b68` | challenger |

- Piper runs through the project API with `noise_scale=0` and
  `noise_w_scale=0`.
- MMS uses Transformers VITS on CPU. Its stochastic duration predictor is made
  reproducible with a deterministic per-row seed derived from the frozen row
  hash.
- Both waveforms are converted to mono float32 at the project sample rate
  before the same NeMo service receives them.
- One fixed excluded sentence may warm each backend before timed panel rows.

## Metrics and validity

Every source text remains in the denominator. Empty ASR output is a valid
system output and receives full-deletion error counts. A synthesis or ASR
exception also receives full-deletion counts and makes the arm mechanically
invalid.

Report per arm:

- micro/macro CER and WER, edit counts, failures, and empty hypotheses;
- full-utterance synthesis latency and real-time factor (p50/p95/max);
- output duration, RMS, peak, finiteness, and output hashes;
- per-channel CER/WER; and
- model/runtime identity.

This implementation does not measure true streaming first-audio latency.

## Frozen development rule

The challenger advances only when both arms complete all 40 rows and all are
true:

- absolute micro-CER reduction is at least 0.02;
- absolute micro-WER reduction is nonnegative;
- paired session-bootstrap 95% upper bound for
  `MMS CER - Piper CER` is below zero; and
- MMS p50 full-render real-time factor is at most 1.0.

If the rule fails, retain Piper and do not open final. If it passes, compare
both arms once on the locked final panel. Automatic confirmation requires
positive CER improvement, nonworse WER, and a paired 95% CER-difference upper
bound below zero. Even then, production promotion is provisional until human
Persian listening is performed and the CC-BY-NC deployment boundary is
acceptable.

## Commands

```bash
.venv-tts/bin/python scripts/evaluate_tts_candidates_v1.py --stage plan
.venv-tts/bin/python scripts/evaluate_tts_candidates_v1.py --stage development
```

Run `--stage final` only when the development receipt unlocks it.
