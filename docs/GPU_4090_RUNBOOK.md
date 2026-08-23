# RTX 4090 runbook

The 4090 is a physical 24 GB card and should satisfy the thesis hardware-size condition if the supervisor confirms it. It enables official live evaluation; it does not make component-only or synthetic timings official.

## 1. Preflight

Use explicit executables; terminal auto-activation is disabled for this workspace.

```bash
cd /mnt/md0/mehbakh/Thesis_Project
.venv/bin/python -m thesis_s2s.cli gpu-preflight \
  --out results/hardware/4090_preflight.json
.venv/bin/python -m thesis_s2s.cli verify-upstreams
```

Confirm in the JSON:

- device name contains `4090` and physical VRAM is in the 12–24 GB interval;
- CUDA is available and BF16 is supported;
- no `memory_capped: true` proxy is used;
- upstream commits match `third_party/UPSTREAMS.lock.json`;
- the conversation audit is present and passes before adaptation is attempted.

## 2. Prepare conversational supervision

```bash
# CSV planning is fast and reproducible; full preparation is resumable I/O.
.venv/bin/python -m thesis_s2s.cli plan-conversation-corpus
.venv/bin/python -m thesis_s2s.cli prepare-conversation-episodes
.venv/bin/python -m thesis_s2s.cli normalize-conversation-rights-metadata
.venv/bin/python -m thesis_s2s.cli audit-prepared-episodes

# Start the existing offline Community-1 service on the 4090.
cd /mnt/md0/mehbakh/asr_nemo_soroush
docker compose --profile gpu up -d diarization-gpu
diar_ip=$(docker inspect asr_nemo_soroush_diarization \
  --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
curl -f "http://${diar_ip}:8081/health/ready"
cd /mnt/md0/mehbakh/Thesis_Project
export DIARIZATION_SERVICE_URL="http://${diar_ip}:8081"
export DIARIZATION_TIMEOUT_SECONDS=600
.venv/bin/python -m thesis_s2s.cli diarize-conversation-episodes
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes

.venv/bin/python -m thesis_s2s.cli sample-conversation-qa
# A reviewer fills conversation_manual_qa.csv before the next command.
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa
# The completed supervisor-authorization CSV already exists locally; do not overwrite it.
# In a fresh environment only, create it once and reproduce the documented decision.
# .venv/bin/python -m thesis_s2s.cli create-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli apply-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes \
  --manifest data/processed/manifests/conversation_episode_windows_approved.jsonl
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_approved.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli export-omni2-data
```

The selector's planning gate is not sufficient evidence. Do not start a long adaptation unless both `results/diarized_episode_audit.json` and `results/conversation_audit.json` pass. Episode-level train/validation/test separation, response audio, reference alignment, and manual QA are mandatory. See `YOUTUBE_CONVERSATION_PIPELINE.md`.

Manual review validates labels, not data rights. Supervisor-approved internal training is now documented and machine-audited separately; the source-license gate intentionally remains false and raw-data redistribution remains prohibited.

## 3. Direct-model boundary

The required training is response learning, not another ASR fine-tune. Each example contains a user-turn waveform as input and the next different speaker turn as the assistant target. A valid trainer must optimize assistant response text and assistant speech/audio tokens; it must not reconstruct the user waveform or predict the user transcript as its final task. The practical 24 GB plan is to freeze the speech encoder and most of the language/speech backbones, train the speech projector plus LoRA adapters and speech-output head in BF16 with gradient checkpointing, and preserve session-isolated train/validation/test splits. The barge-in classifier is a separate small supervised training run over interruption/backchannel/noise labels.

`configs/llama_omni2_4090.yaml` is a reviewed preparation contract, not a working trainer. The pinned official LLaMA-Omni2 repository lacks the complete trainer needed for assistant speech-token supervision. Before any adaptation is called successful, a reviewed implementation must prove all acceptance tests listed in that config.

Never use `train-s2s --allow-experimental` as the direct-model result: it is an input reconstruction ablation and its checkpoints are deliberately runtime-ineligible.

## 4. Live official evaluation without raw-audio retention

For each participant use a unique session ID and stable pseudonymous speaker ID:

```bash
.venv/bin/python -m thesis_s2s.cli serve --study --retention features \
  --host 127.0.0.1 --port 8765 \
  --session-id MOS01 --speaker-id P01 --age-bin under_60
```

For an elderly participant use `--age-bin 60plus`. The browser must generate `playback_started` and `playback_stopped_ack`; server-only timings are diagnostic. Use Piper or a validated neural talker, never the formant fallback, for rated sessions.

After all sessions:

```bash
.venv/bin/python -m thesis_s2s.cli export-recordings
.venv/bin/python -m thesis_s2s.cli train-bargein \
  --recorded-jsonl data/processed/manifests/recorded.jsonl
.venv/bin/python -m thesis_s2s.cli study-summary \
  --out results/eval/human_study.json
```

Feature mode writes no WAV. Metrics-only mode is available if acoustic aggregates are also prohibited, but it cannot supply feature-based detector training evidence.

## 5. Report and freeze

Report p50, p95, and maximum first-audio and barge-in values until the supervisor resolves the 500 ms wording. Keep component and live-client results in separate tables.

```bash
.venv/bin/ruff check src tests
.venv/bin/mypy src
.venv/bin/python -m pytest -q
.venv/bin/python -m thesis_s2s.cli release-snapshot \
  --out results/release/4090_snapshot.json
sha256sum results/hardware/4090_preflight.json \
  results/eval/human_study.json results/release/4090_snapshot.json
```

Archive the exact commit, environment snapshot, model hashes, dataset audit, upstream lock, and live evidence JSON together. Do not publish restricted/raw datasets.
