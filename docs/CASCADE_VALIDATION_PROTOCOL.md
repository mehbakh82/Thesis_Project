# Frozen real-service cascade validation protocol

Status: frozen before any validation-row read or execution.

## Purpose

The cascade v4 working-system result passed all nine rows in its disclosed
train-only development panel. This follow-up evaluates the unchanged runtime on
a source-session-group-isolated validation split. It is a one-time execution
check, not a new tuning round.

## Frozen inputs

- Parent result:
  `results/eval/cascade_real_service_train_panel_v4.json`, SHA-256
  `d6c166c7bb8d93f332c277bdb4ac1fef5bfdfe080985025d7ec0072cda738b36`.
- Validation manifest: `data/processed/moshi_finetune/val.jsonl`, exactly
  131 rows, SHA-256
  `a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7`.
- Export audit: `results/moshi_export_audit.json`, SHA-256
  `9f4fdf6301ed7d63713df6e4818c41e0a4bfefbe0a14e88e0faaba6ef6d5ee00`.
- Fixed floor-spaced row indices: `0, 16, 32, 48, 65, 81, 97, 113, 130`.
- Components, revisions, decoding, thresholds, and retry bound are unchanged
  from cascade v4.
- The independent export audit declares channel 0 as assistant and channel 1 as
  user. This evaluator therefore sends channel 1 to ASR. The earlier v4
  train-panel evaluator sent channel 0; its 9/9 result remains a real
  component-chain smoke test but is not treated as a proper user-turn test.
- Entrypoint: `.venv/bin/python scripts/evaluate_cascade_validation.py`.
- Write-once output:
  `results/eval/cascade_real_service_validation_panel.json`.

The evaluator may read only the nine declared validation rows. It must never
open a test manifest. Source transcripts and generated reply strings are not
retained; only hashes, script statistics, audio statistics, component identity,
and pass/fail fields are written.

## Acceptance

Every row must pass the unchanged v4 gates:

- source audio hash matches the manifest and audited user channel 1 enters ASR;
- NeMo ASR succeeds, produces at least four characters, and at least 80% of
  letters use Arabic/Persian script;
- the exact cached Qwen responder initializes and generates without rule
  fallback, using one attempt or the already-frozen maximum of two;
- the reply contains at least four characters and at least 80% Arabic/Persian-
  script letters;
- exact Piper synthesis returns finite normalized audio, at least 1,600
  samples, with RMS at least `0.001`.

Overall passage also requires NeMo health HTTP 200, all nine fixed indices, the
exact parent result and export audit, the audited 131-row validation count with
zero group-split leaks, and no final-test access.

## Decision and claims

This protocol is one-time and fail-closed. No prompt, threshold, row, decoder,
or fallback may be revised from its result.

A pass supports: “The frozen real-service NeMo → Qwen → Piper cascade produced
Persian-script text and non-silent speech on all nine predeclared rows from a
session-group-isolated validation split.”

Even a pass does not establish semantic relevance, naturalness, human task
success, population-level generalization, physical 12–24 GB deployment, or
official browser `T_first_audio`. Full-turn generation time is descriptive
only. The direct Moshi experiments and their final-test firewalls are
unaffected.
