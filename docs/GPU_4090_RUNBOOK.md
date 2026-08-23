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
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/full_diarized.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli export-omni2-data
```

Do not start a long adaptation if `results/conversation_audit.json` fails. Session-level train/validation/test separation and response audio are mandatory.

## 3. Direct-model boundary

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
