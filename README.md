# Persian full-duplex speech-to-speech (BSc thesis)

An evidence-first Persian speech prototype that keeps the microphone active during assistant playback and can stop the real browser audio source on a classical barge-in decision.

## Current status

| Requirement | Honest status |
|---|---|
| Working spoken conversation | Implemented as NeMo Persian ASR → local Qwen2.5-0.5B (rule fallback) → Piper Persian TTS |
| Full-duplex control | Continuous browser PCM16 stream, rolling energy/F0/MFCC detector, server stop event, client stop acknowledgement |
| Direct speech LLM | **Not implemented**. The existing 7 MB artifact is an input-mel reconstruction experiment and is fail-closed at runtime |
| 100–200 h conversation candidates | Full five-source inventory: **775.887 h / 1,442 long episodes**. Deterministic whole-episode plan: **180.043 h / 287 episodes**, four channels, largest share 51.36% |
| Conversational interruption supervision | Episode preparation, diarization alignment, response-pair builder, and fail-closed audits are implemented. The selected hours remain candidates until the 4090 diarization run and manual QA |
| Barge-in >80% | Met only on harmonic synthetic held-out data; real speaker/session-held-out evidence is pending |
| ≤500 ms and 12–24 GB official test | Pending live browser measurements on a physical 12–24 GB GPU |
| Human study | Incomplete; 5–10 participants and at least two aged 60+ are still required. Raw WAV retention is optional |

The machine-generated evidence is in `results/corpus_audit.json` and `results/eval/human_study.json`. Neither synthetic latency nor an H100 memory cap is accepted as official end-to-end evidence.

## Architecture

```text
continuous 16 kHz microphone ─┬─> rolling energy/F0/MFCC barge-in ─> stop browser source
                              └─> NeMo ASR ─> Qwen/rules ─> Piper ─> full PCM reply
```

Serving is deliberately cascade-only. A future direct model requires both a separately implemented/tested runtime loader and a validated `deployable_s2s_v1` artifact; checkpoint metadata alone cannot activate it.

## Setup

```bash
cd /mnt/md0/mehbakh/Thesis_Project
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e ".[dev]"
export PYTHONPATH=src
```

Supply S3 credentials only through `S3_*` environment variables (see `.env.example`). They are passed to rclone through its subprocess environment, never command arguments.
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
.venv/bin/python -m thesis_s2s.cli sample-conversation-qa
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa
.venv/bin/python -m thesis_s2s.cli create-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli apply-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli audit-corpus --no-check-files
.venv/bin/python -m thesis_s2s.cli audit-alignment
.venv/bin/python -m thesis_s2s.cli build-conversations --in-jsonl data/processed/manifests/conversation_episode_windows_approved.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli serve --study --retention features
.venv/bin/python -m thesis_s2s.cli export-recordings
.venv/bin/python -m thesis_s2s.cli study-summary
.venv/bin/python -m thesis_s2s.cli gpu-preflight
.venv/bin/python -m thesis_s2s.cli release-snapshot
.venv/bin/python -m thesis_s2s.cli eval --path both
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
```

`train-s2s` is retained only for ablation work and refuses to run unless `--allow-experimental` is supplied. It must not be cited as a trained end-to-end S2S model.

## Metric contract

- `T_first_audio`: end-of-speech to browser-scheduled first audio, reported by the client.
- `T_barge_in`: acoustic onset estimate to confirmed browser source stop, reported by the client.
- Report p50, p95, and maximum. Whether the definition's 500 ms wording binds max, p95, or p50 is an explicit supervisor decision; the historical p50 gate is provisional.
- Official latency requires live client telemetry and a physical 12–24 GB GPU.
- Primary barge-in evidence requires speaker/session-held-out real interactions; lossy aggregate features may be used without retaining WAV, subject to ethics approval.
- Human evaluation requires 5–10 Persian speakers, at least two aged 60+, with complete ratings.

See `docs/YOUTUBE_CONVERSATION_PIPELINE.md`, `docs/METRICS.md`, `docs/HUMAN_STUDY.md`, `docs/NO_RECORDING_ALTERNATIVES.md`, `docs/GPU_4090_RUNBOOK.md`, `docs/MANUAL_QA_FA.md`, `docs/RIGHTS_REVIEW_FA.md`, `docs/SUPERVISOR_DECISIONS.md`, and `docs/REVIEW.md`.
