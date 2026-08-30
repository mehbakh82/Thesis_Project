# Persian full-duplex speech-to-speech (BSc thesis)

An evidence-first Persian speech prototype that keeps the microphone active during assistant playback and can stop the real browser audio source on a classical barge-in decision.

## Current status

| Requirement | Honest status |
|---|---|
| Working spoken conversation | Implemented as NeMo Persian ASR → local Qwen2.5-0.5B (rule fallback) → Piper Persian TTS |
| Full-duplex control | Continuous browser PCM16 stream, rolling energy/F0/MFCC detector, server stop event, client stop acknowledgement |
| Direct speech LLM | V1 remains rejected for English drift/near-silence. V2 completed all 2,000 steps but failed the corrected complete-prompt official-runtime gates (best: step 400, 2/9). The bounded embedding-preserving v3 completed all 500 steps; complete-scope validation loss improved from 2.084004 to 1.879258, but its five candidates passed 0/9, 0/9, 0/9, 1/9, and 0/9 runtime rows. V3 therefore also failed closed. The fresh 14-session/11.894-hour final test remains untouched and no direct adapter is deployment-eligible. |
| 100–200 h conversation corpus | Full inventory: **775.887 h / 1,442 long episodes**. The production selection is **219.946 candidate h / 309 episodes** across four channels; 1,129 windows contain **207.154 automatically classified multi-speaker h**, **242.445 aligned staging h**, and **6,754 non-reused response pairs / 123.796 source-pair h**. The immutable Piper derivative is **108.584 measured stereo h**, inside the formal band |
| Conversational interruption supervision | Raw speaker boundaries recover **770 conservative candidates** (717 interruption-like, 53 backchannel-like) from the 6,754 pairs. The preserved deterministic 24-row sheet was sampled from the earlier 712-candidate pool and covers all four channels / **168.3 seconds** of excerpt audio. They remain automatic candidates; **zero human-verified direct interruptions** are claimed under the waiver |
| Barge-in >80% | Met only on harmonic synthetic held-out data; real speaker/session-held-out evidence is pending |
| ≤500 ms and 12–24 GB official test | Pending live browser measurements on a physical 12–24 GB GPU |
| Human study | Incomplete; 5–10 participants and at least two aged 60+ are still required. Raw WAV retention is optional |

Machine-generated corpus evidence is in `results/diarized_episode_audit_combined_authorized.json`, `results/conversation_yield_estimate_combined.json`, and `results/corpus_audit.json`; study evidence is in `results/eval/human_study.json`. Neither synthetic latency nor an H100 memory cap is accepted as official end-to-end evidence.

## Architecture

```text
continuous 16 kHz microphone ─┬─> rolling energy/F0/MFCC barge-in ─> stop browser source
                              └─> NeMo ASR ─> Qwen/rules ─> Piper ─> full PCM reply
```

The bounded v3 experiment restored the pinned trainer's rank-128,
embedding-frozen LoRA default while keeping v2 data, context, optimizer, seed,
and output gates unchanged. Training, complete-scope reevaluation, and the
official-runtime panel completed for steps 100–500. No candidate passed all
nine rows, so selection returned null and the final-test firewall remained
closed.

The project server remains deliberately cascade-only until a Persian adapter passes every gate. The v1 adapter loads in Kyutai's pinned official server but fails autoregressive Persian output, so it cannot activate the direct path. Checkpoint metadata alone can never activate either that failed adapter or the legacy reconstruction artifact.

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
.venv/bin/python -m thesis_s2s.cli audit-moshi-data
.venv/bin/python -m thesis_s2s.cli serve --study --retention features
.venv/bin/python -m thesis_s2s.cli export-recordings
.venv/bin/python -m thesis_s2s.cli study-summary
.venv/bin/python -m thesis_s2s.cli gpu-preflight
.venv/bin/python -m thesis_s2s.cli release-snapshot
.venv/bin/python scripts/audit_moshi_dependencies.py
.venv/bin/python scripts/prepare_moshi_v2_splits.py
.venv/bin/python scripts/record_moshi_profile_probe.py --run-dir checkpoints/moshi_h100_100s_profile_probe --probe-config configs/moshi_h100_100s_profile_probe.yaml --full-config configs/moshi_h100_v2.yaml --out results/hardware/moshi_h100_100s_profile_probe.json
.venv-moshi/bin/python scripts/reevaluate_moshi_checkpoints.py --training-config configs/moshi_h100_v2.yaml --run-dir checkpoints/moshi_fa_100s_v2 --metrics-out checkpoints/moshi_fa_100s_v2/metrics.reeval.jsonl --report-out results/moshi_v2_validation_reevaluation.json --heldout-test-manifest data/processed/moshi_finetune_v2/test.jsonl
.venv-moshi/bin/python scripts/evaluate_moshi_v2_runtime_candidates.py
.venv/bin/python scripts/select_moshi_v2_checkpoint.py
.venv-moshi/bin/python scripts/validate_moshi_adapter.py --selection results/moshi_v2_checkpoint_selection.json --training-config configs/moshi_h100_v2.yaml --out results/moshi_v2_adapter_validation.json --runtime-device cuda
.venv-moshi/bin/python scripts/evaluate_moshi_v2_final_test.py
.venv-moshi/bin/python scripts/evaluate_moshi_v2_final_runtime.py
.venv-moshi/bin/python scripts/record_moshi_v2_training_run.py
.venv/bin/python scripts/run_moshi_v2_posttraining.py
.venv/bin/python scripts/run_moshi_v3_posttraining.py
.venv-moshi/bin/python scripts/record_moshi_v3_training_run.py

# Frozen v4 protocol: rank-128 LoRA plus exactly two text embeddings.
# Run preflight only from a clean, committed worktree; the probe and training
# launches additionally require MOSHI_TEXT_EMBEDDINGS_ONLY=1.
.venv/bin/python scripts/preflight_moshi_v4.py
.venv/bin/python scripts/record_moshi_profile_probe.py --run-dir checkpoints/moshi_h100_100s_v4_text_embed_probe --probe-config configs/moshi_h100_v4_probe.yaml --full-config configs/moshi_h100_v4.yaml --embedding-policy configs/moshi_v4_embedding_policy.json --out results/hardware/moshi_h100_v4_profile_probe.json
.venv/bin/python scripts/run_moshi_v4_posttraining.py --training-invocation-id <systemd-invocation-id> --launch-commit <training-launch-commit>
.venv-moshi/bin/python scripts/record_moshi_v4_training_run.py --launch-commit <training-launch-commit> --service-unit <systemd-unit> --invocation-id <systemd-invocation-id>
# Only after an eligible schema-6 selection and certificate are committed:
.venv/bin/python scripts/run_moshi_v4_final.py

# Historical v1 evidence
.venv-moshi/bin/python scripts/reevaluate_moshi_checkpoints.py
.venv/bin/python scripts/select_moshi_checkpoint.py
.venv-moshi/bin/python scripts/validate_moshi_adapter.py --runtime-device cuda
.venv-moshi/bin/python scripts/evaluate_moshi_adapter.py
.venv/bin/python scripts/record_moshi_training_run.py
.venv-moshi/bin/python scripts/validate_moshi_server_runtime.py
.venv/bin/python -m thesis_s2s.cli eval --path both
.venv/bin/ruff check src tests scripts
.venv/bin/python -m pytest -q
```

Conversation commands are strict by default. The active time-constrained,
automatic-only training path must pass the same explicit
`--qa-waiver configs/conversation_qa_waiver.yaml` to pair building, audit, and
Moshi export. See `docs/QA_WAIVER.md`; the waiver never creates human-verified
or strict-coverage evidence.

`gpu-preflight` reports H100 hardware, the isolated Moshi stack, strict and
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
`docs/MOSHI_SELECTION_PROTOCOL.md`, `docs/MOSHI_V2_SELECTION_PROTOCOL.md`, `docs/MOSHI_V2_PROTOCOL_CORRECTION.md`, `docs/MOSHI_V3_SELECTION_PROTOCOL.md`, `docs/QA_WAIVER.md`,
`docs/REPOSITORY_SECURITY.md`, `docs/METRICS.md`, `docs/HUMAN_STUDY.md`,
`docs/NO_RECORDING_ALTERNATIVES.md`, `docs/GPU_4090_RUNBOOK.md`,
`docs/MANUAL_QA_FA.md`, `docs/INTERRUPTION_QA_FA.md`,
`docs/RIGHTS_REVIEW_FA.md`, `docs/SUPERVISOR_DECISIONS.md`, and
`docs/REVIEW.md`.

The ordered evidence and delivery path to a fully complete project is tracked
in `docs/PROJECT_10_OF_10_CHECKLIST.md`.
