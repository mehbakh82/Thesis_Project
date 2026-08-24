# Conversation QA waiver and limited-training policy

Status: acknowledged on 2026-08-24 by the student/project owner.

## Decision and reason

Best practice for this corpus is to complete both prepared listening reviews
before model training: 40 full-window rows (about 6.9 reviewer-hours at one
real-time pass plus the documented checking allowance) and 24 interaction
excerpts (168.3 seconds of audio). The student cannot complete or delegate the
reviews within the available schedule and cannot wait for a separate written
scope response. The project therefore proceeds with a documented automatic-only
QA waiver for the limited internal training run.

This is a student methodological decision. It does **not** claim that the
supervisor approved waiving QA. The already recorded supervisor approval for
internal training on the crawled YouTube corpus remains the data-use basis; the
two decisions are separate.

The machine-readable decision is
`configs/conversation_qa_waiver.yaml`. Its SHA-256 is copied into every pair and
validated again by the audit, Moshi exporter, profile probe, and preflight. A
changed, missing, invalid, or mismatched waiver fails closed.

## What the waiver permits

- Build non-reused natural user/response pairs from the authorized,
  automatically diarized and caption-aligned windows.
- Retain automatic overlap/interruption candidates as pseudo-label metadata.
- Export and train the pinned Moshi/Moshika adaptation for limited internal
  research when all non-human data, authorization, file, duration, split,
  voice, and reproducibility gates pass.
- Report results explicitly as obtained under automatic-label QA waiver.

The operative readiness field is `training_ready_under_qa_waiver=true`.
`gpu-preflight` reports the selected policy and keeps strict readiness beside
it rather than replacing it.

## What the waiver does not permit

- No row may be called human-verified merely because QA was waived.
- No automatic interaction candidate may be called a verified interruption or
  backchannel.
- `thesis_coverage_ok`, `final_training_ready`, and strict 10/10 readiness stay
  false unless the original human requirements are actually satisfied.
- The waiver is not consent, copyright permission, source-license
  verification, redistribution permission, human-study evidence, model-quality
  evidence, or target-4090 evidence.
- Blank QA cells are never filled, inferred, or promoted by code.

## Preserved best-practice artifacts

The following remain unchanged and usable for later review:

- `data/processed/manifests/conversation_manual_qa.csv`;
- `data/processed/manifests/conversation_interruption_qa.csv`;
- `docs/MANUAL_QA_FA.md`;
- `docs/INTERRUPTION_QA_FA.md`;
- `scripts/review_interaction_candidate.py`.

The preflight waiver path checks that these artifacts still exist. If time or a
reviewer later becomes available, complete and apply the sheets, rebuild
without `--qa-waiver`, and return to the strict path.

## Limited automatic-only command path

Use the already authorized combined manifest directly; do not run either
`apply-*-qa` command against blank sheets.

```bash
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl \
  --qa-waiver configs/conversation_qa_waiver.yaml

.venv/bin/python -m thesis_s2s.cli audit-conversations \
  --qa-waiver configs/conversation_qa_waiver.yaml

.venv/bin/python -m thesis_s2s.cli export-moshi-data \
  --assistant-audio-mode piper \
  --qa-waiver configs/conversation_qa_waiver.yaml

.venv/bin/python -m thesis_s2s.cli gpu-preflight \
  --out results/hardware/current_preflight.json
```

Proceed to the exact-shape memory probe and training only if both the effective
waiver gate and all non-QA training gates pass. The strict fields are expected
to remain false.

## Required thesis disclosure

Use language equivalent to the following in methods and limitations:

> Best-practice stratified listening QA was designed and the review sheets,
> criteria, and playback tooling were prepared. Because the student could not
> complete or delegate the estimated review effort within the project schedule,
> model adaptation used conservative automatic diarization/alignment labels
> under a documented student-authorized waiver. Automatic interaction labels
> were treated as pseudo-labels, not human-verified interruption evidence.
> Consequently, strict human-QA corpus coverage is not claimed, and this is a
> limitation of the training data and resulting evaluation.

Report the actual counts and failures from the frozen audit/export alongside
this disclosure. Do not phrase the waiver as proof that QA was unnecessary.

## Restoring the strict path

1. Complete both preserved sheets with identified reviewers.
2. Apply window QA and interaction QA fail-closed.
3. Rebuild and audit without `--qa-waiver`.
4. Export without `--qa-waiver` and require
   `final_training_ready=true`.
5. Re-run preflight and require `strict_training_data_ready=true`.

This recovery path requires no regeneration of the sampling sheets or reviewer
guides.
