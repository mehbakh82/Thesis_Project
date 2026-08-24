# Persian full-duplex speech-to-speech (BSc thesis)

An evidence-first Persian speech prototype that keeps the microphone active during assistant playback and can stop the real browser audio source on a classical barge-in decision.

## Current status

| Requirement | Honest status |
|---|---|
| Working spoken conversation | Implemented as NeMo Persian ASR → local Qwen2.5-0.5B (rule fallback) → Piper Persian TTS |
| Full-duplex control | Continuous browser PCM16 stream, rolling energy/F0/MFCC detector, server stop event, client stop acknowledgement |
| Direct speech LLM | Official Moshi LoRA trainer, pinned runtime, Persian stereo-response exporter, isolated environment, and H100 profiles are implemented and tested; all three exact-revision base blobs pass byte/SHA-256 verification. A real one-step H100 smoke passed end to end (loss 4.614331, 15.258 GB peak). Full adapter training is gated by the window/pair listening QA, final export, an exact-shape one-step memory probe, and adequate shared-H100 headroom. The old 7 MB reconstruction artifact stays runtime-ineligible |
| 100–200 h conversation candidates | Full inventory: **775.887 h / 1,442 long episodes**. The measured final plan is **196.546 candidate h / 296 episodes** across four channels; 1,021 windows contain **181.824 automatically verified multi-speaker h** and an estimated **6,017 response pairs / 105.727 pair h** |
| Conversational interruption supervision | Raw speaker boundaries recover **712 conservative candidates** (669 interruption-like, 43 backchannel-like) from the 6,017 pairs. A deterministic 24-row sheet covers all four channels and only **168.3 seconds** of excerpt audio. They remain automatic candidates; **zero human-verified direct interruptions** are claimed until that listening review passes |
| Barge-in >80% | Met only on harmonic synthetic held-out data; real speaker/session-held-out evidence is pending |
| ≤500 ms and 12–24 GB official test | Pending live browser measurements on a physical 12–24 GB GPU |
| Human study | Incomplete; 5–10 participants and at least two aged 60+ are still required. Raw WAV retention is optional |

Machine-generated corpus evidence is in `results/diarized_episode_audit_combined_authorized.json`, `results/conversation_yield_estimate_combined.json`, and `results/corpus_audit.json`; study evidence is in `results/eval/human_study.json`. Neither synthetic latency nor an H100 memory cap is accepted as official end-to-end evidence.

## Architecture

```text
continuous 16 kHz microphone ─┬─> rolling energy/F0/MFCC barge-in ─> stop browser source
                              └─> NeMo ASR ─> Qwen/rules ─> Piper ─> full PCM reply
```

The project server remains deliberately cascade-only until the Persian adapter passes its gates. The selected direct path is trained on the H100 and loaded by Kyutai's pinned official Moshi streaming server; checkpoint metadata alone can never activate the legacy reconstruction artifact.

```text
natural Persian user audio + approved next-turn text
        -> deterministic single-voice Persian Piper target
        -> official Moshi text + Mimi assistant-token LoRA training on H100
        -> official streaming Moshi runtime (4090 used only for final fit/latency)
```

## Setup

```bash
cd Thesis_Project
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
export PYTHONPATH=src
```

Use either a protected preconfigured rclone remote via `THESIS_RCLONE_REMOTE` or ephemeral `S3_*` environment variables (see `.env.example`). Credentials are never copied from rclone configuration, placed in the repository, or passed as command arguments.
If a key has ever been stored in a plaintext cheatsheet or exposed in output,
rotate it and replace the document with environment-variable placeholders.

## Core commands

```bash
.venv/bin/python -m thesis_s2s.cli plan-conversation-corpus
.venv/bin/python -m thesis_s2s.cli prepare-conversation-episodes
.venv/bin/python -m thesis_s2s.cli normalize-conversation-rights-metadata
.venv/bin/python -m thesis_s2s.cli audit-prepared-episodes
.venv/bin/python -m thesis_s2s.cli diarize-conversation-episodes
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes
.venv/bin/python -m thesis_s2s.cli annotate-conversation-noise
.venv/bin/python -m thesis_s2s.cli estimate-conversation-yield
.venv/bin/python -m thesis_s2s.cli select-conversation-reserve
.venv/bin/python -m thesis_s2s.cli sample-conversation-qa
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa
.venv/bin/python -m thesis_s2s.cli sample-interruption-qa
.venv/bin/python -m thesis_s2s.cli apply-interruption-qa
.venv/bin/python -m thesis_s2s.cli create-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli apply-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli audit-corpus --no-check-files
.venv/bin/python -m thesis_s2s.cli audit-alignment
.venv/bin/python -m thesis_s2s.cli build-conversations --in-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli export-moshi-data --assistant-audio-mode piper
.venv/bin/python -m thesis_s2s.cli serve --study --retention features
.venv/bin/python -m thesis_s2s.cli export-recordings
.venv/bin/python -m thesis_s2s.cli study-summary
.venv/bin/python -m thesis_s2s.cli gpu-preflight
.venv/bin/python -m thesis_s2s.cli release-snapshot
.venv/bin/python -m thesis_s2s.cli eval --path both
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pytest -q
```

Conversation commands are strict by default. The active time-constrained,
.venv/bin/python -m thesis_s2s.cli audit-moshi-data
automatic-only training path must pass the same explicit
`--qa-waiver configs/conversation_qa_waiver.yaml` to pair building, audit, and
Moshi export. See `docs/QA_WAIVER.md`; the waiver never creates human-verified
or strict-coverage evidence.

`gpu-preflight` reports H100 hardware, the isolated Moshi stack, strict and
.venv/bin/python scripts/select_moshi_checkpoint.py
waiver-limited data/export readiness, current full-profile memory headroom, and
physical 4090 evaluation as separate gates. A 4090 is never required to train
the adapter.

`train-s2s` is retained only for ablation work and refuses to run unless `--allow-experimental` is supplied. It must not be cited as a trained end-to-end S2S model.

## Metric contract

- `T_first_audio`: end-of-speech to browser-scheduled first audio, reported by the client.
- `T_barge_in`: acoustic onset estimate to confirmed browser source stop, reported by the client.
- Report p50, p95, and maximum. Whether the definition's 500 ms wording binds max, p95, or p50 is an explicit supervisor decision; the historical p50 gate is provisional.
- Official latency requires live client telemetry and a physical 12–24 GB GPU.
- Primary barge-in evidence requires speaker/session-held-out real interactions; lossy aggregate features may be used without retaining WAV, subject to ethics approval.
- Human evaluation requires 5–10 Persian speakers, at least two aged 60+, with complete ratings.

See `docs/YOUTUBE_CONVERSATION_PIPELINE.md`, `docs/MOSHI_H100_RUNBOOK.md`,
`docs/MOSHI_SELECTION_PROTOCOL.md`, `docs/QA_WAIVER.md`,
`docs/REPOSITORY_SECURITY.md`, `docs/METRICS.md`, `docs/HUMAN_STUDY.md`,
`docs/NO_RECORDING_ALTERNATIVES.md`, `docs/GPU_4090_RUNBOOK.md`,
`docs/MANUAL_QA_FA.md`, `docs/INTERRUPTION_QA_FA.md`,
`docs/RIGHTS_REVIEW_FA.md`, `docs/SUPERVISOR_DECISIONS.md`, and
`docs/REVIEW.md`.

The ordered evidence and delivery path to a fully complete project is tracked
in `docs/PROJECT_10_OF_10_CHECKLIST.md`.
