# Frozen supplemental Shenava ASR screen v1

**Frozen:** 2026-09-08, before Shenava touched the project development panel

**Evidence class:** post-hoc candidate screen; not a predeclared final-model
selection experiment

## Why this exists

Shenava-Koochik v1.0 was discovered after the three-arm ASR v1 development
comparison completed. Its published Persian specialization, compact 114M
architecture, Apache-2.0 release, and cross-platform runtime make it important
to screen, but retroactively adding it to the v1 protocol would falsify that
protocol's chronology.

This supplement therefore reuses only the already-open v1 development panel
and clearly labels the analysis post-hoc. It never accesses the sealed v1 final
panel. A Shenava win can justify a new, independently frozen confirmation
panel; it cannot by itself promote the model.

## Frozen identities and inputs

- Source manifest SHA-256:
  `aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86`
- ASR v1 plan SHA-256:
  `ca73a72c5f27b9459ccdf4298c282fb153af3b03c2709dcb23bbb89bb542f0ef`
- Panel: the exact ASR v1 development panel: 40 rows, eight sessions, 20
  Digiato and 20 Zoomit rows, 272.465 seconds.
- Incumbent: the exact live fine-tuned NeMo service images already pinned by
  ASR v1.
- Challenger: `Reza2kn/Shenava-Koochik-v1.0-sherpa-onnx`, revision
  `f063f38cb38fe02df39887ac65c441b755ab25a2`; model ONNX SHA-256
  `6a564b5541920ce1c37bbc91d22e4b3a6838648b9b327eb88997e8db1f90950d`.
- Runtime: `sherpa-onnx==1.13.7`, four CPU threads, no Persian ITN.

The plan stage records the exact protocol, evaluator, model-file, runtime,
source, v1-plan, and NeMo identities before inference.

## Metrics and denominator

Both arms receive identical audio and the same automatic reference captions.
Report micro/macro CER and WER, failures, empty hypotheses, RTF, per-channel
metrics, and a paired 10,000-draw session-bootstrap interval for
`Shenava CER - NeMo CER`.

Every row remains in the denominator. Empty output receives full-deletion
counts. Runtime exceptions also receive full-deletion counts and invalidate the
arm. Receipts retain hashes and counts only, never transcript text.

## Screen decision

Shenava advances to a new confirmation protocol only if:

- both arms complete all 40 rows;
- absolute micro-CER reduction is at least 0.02;
- micro-WER is nonworse; and
- the paired CER-difference 95% upper bound is below zero.

Otherwise NeMo remains selected and no final/confirmation panel is opened.
Official benchmark numbers are context only and never substitute for this
project-distribution result. Training overlap between Shenava's VisualEars
corpus and these YouTube sources is not established, so even a win would retain
that limitation until provenance is verified.

## Commands

```bash
.venv-shenava/bin/python scripts/evaluate_asr_shenava_supplemental_v1.py --stage plan
.venv-shenava/bin/python scripts/evaluate_asr_shenava_supplemental_v1.py --stage screen
```
