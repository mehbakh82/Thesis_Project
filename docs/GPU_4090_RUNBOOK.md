# RTX 4090 runbook

The 4090 is a physical 24 GB card and should satisfy the thesis hardware-size condition if the supervisor confirms it. It enables official live evaluation; it does not make component-only or synthetic timings official.

## 1. Preflight

Use explicit executables; terminal auto-activation is disabled for this workspace.

```bash
cd Thesis_Project
.venv/bin/python -m thesis_s2s.cli gpu-preflight \
  --out results/hardware/4090_preflight.json
.venv/bin/python -m thesis_s2s.cli verify-upstreams
```

Confirm in the JSON:

- device name contains `4090` and physical VRAM is in the 12–24 GB interval;
- CUDA is available and BF16 is supported;
- no `memory_capped: true` proxy is used;
- upstream commits match `third_party/UPSTREAMS.lock.json`;
- the H100 environment and one-step wiring reports pass their current hashes;
- `human_review` shows both the 40-row window sheet and 24-row interaction
  sheet as complete and applied fail-closed;
- the conversation audit and Moshi export pass before adaptation is attempted.

`evaluation_hardware_ready` is the 4090 gate. The separate
`adaptation_run_ready` and `adaptation_launch_safe_now` fields describe the
H100 training handoff and do not require the 4090.

## 2. Verify the H100 preparation handoff

Corpus preparation, diarization, QA application, Moshi export, and adaptation
belong on the H100. They are prerequisites for this later 4090 run, not work
that must be repeated on the target card. The H100 sequence is:

```bash
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes
.venv/bin/python -m thesis_s2s.cli annotate-conversation-noise
.venv/bin/python -m thesis_s2s.cli estimate-conversation-yield
.venv/bin/python -m thesis_s2s.cli sample-conversation-qa
# A reviewer fills conversation_manual_qa.csv and the already generated
# conversation_interruption_qa.csv; no new recording is required.
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa
.venv/bin/python -m thesis_s2s.cli apply-interruption-qa
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes \
  --manifest data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli export-moshi-data --assistant-audio-mode piper
MOSHI_DISTRIBUTED_BACKEND=gloo \
  .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
  scripts/moshi_train_entry.py configs/moshi_h100_profile_probe.yaml
.venv/bin/python scripts/record_moshi_profile_probe.py
.venv/bin/python -m thesis_s2s.cli gpu-preflight \
  --out results/hardware/current_preflight.json
```

When the 4090 becomes available, copy or mount the immutable base files,
adapter, checkpoint config, and evidence reports. Confirm their SHA-256 values;
do not re-diarize the corpus or retrain merely because the evaluation GPU
changed. Manual review validates labels, not data rights. Supervisor-approved
internal training is machine-audited separately, while raw-data redistribution
remains prohibited.

## 3. Direct-model boundary

Train the direct model on the H100 before this run. The selected engineering
path is Kyutai's pinned Moshi LoRA trainer with
`configs/moshi_h100.yaml`; see `MOSHI_H100_RUNBOOK.md`. It learns from stereo
user/assistant conversations and supervises assistant text plus Mimi speech
tokens. This is response learning, not another ASR fine-tune.

The 4090 is used here to prove that the resulting base model plus Persian LoRA
adapter fits a physical 24 GB device and meets live browser latency. It does not
need to perform the expensive adaptation itself. Archive the exact Moshi base
revision, adapter hash, configuration, and H100 training logs before copying
the artifact to the 4090.

`configs/llama_omni2_4090.yaml` remains a superseded preparation record because
the pinned LLaMA-Omni2 release lacks the required trainer. Never use
`train-s2s --allow-experimental` as the direct-model result: it reconstructs
the input and its checkpoints are deliberately runtime-ineligible.

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
