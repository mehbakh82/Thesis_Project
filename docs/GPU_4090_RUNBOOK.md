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
- `evaluation_hardware_ready=true` based only on eligible physical hardware
  and CUDA availability;
- the current waiver-bound conversation/export artifacts remain valid, use the
  pinned waiver, and keep every human-verification claim false;
- `human_review` is expected to remain incomplete unless a real reviewer has
  actually completed both preserved sheets. It is not a hardware-readiness
  field and must never be backfilled for the 4090 run.

`evaluation_hardware_ready` is the 4090 hardware gate. The separate
`adaptation_run_ready` and `adaptation_launch_safe_now` fields describe the
historical H100 training handoff and do not require the 4090. Neither field
proves that a deployment-eligible direct adapter exists.

## 2. Verify the completed H100 preparation handoff

Do not re-diarize, rebuild the corpus, re-synthesize the export, or retrain merely
because the evaluation GPU changed. The current submission uses the documented
QA waiver and already has hash-bound conversation/export audits. Verify those
published artifacts and their upstream pins:

```bash
.venv/bin/python scripts/ci_artifact_audit.py
.venv/bin/python -m thesis_s2s.cli verify-upstreams
.venv/bin/python -m thesis_s2s.cli gpu-preflight \
  --out results/hardware/4090_preflight.json
```

The strict manual-QA reconstruction commands remain preserved in
`PROJECT_10_OF_10_CHECKLIST.md`, but they apply only if a real reviewer later
completes both sheets. Never invent reviewer IDs or decisions. A later reviewed
corpus must be versioned and audited separately; it must not overwrite the
frozen waiver lineage.

When the 4090 becomes available, copy or mount only the runtime files and
hash-bound evidence needed for the selected system. Confirm their SHA-256
values. Supervisor-approved internal training is machine-audited separately
from raw-data redistribution, which remains prohibited.

## 3. Direct-model boundary

The current production candidate is the validated NeMo -> Qwen3-4B prompt-v2
-> Piper cascade. It can be run on the 4090 to obtain qualifying physical-card
fit and live-browser latency evidence.

The selected direct-model research path remains the pinned Moshika 7B plus
Moshi-Finetune LoRA. V1-v5 failed their runtime eligibility rules, and v6.2 had
a positive train-only loss signal but passed 0/9 runtime rows for every
candidate. Therefore no direct adapter may be copied, rated, or described as
promoted. If a future experiment produces a validation-eligible adapter, first
archive its exact base revision, adapter hash, configuration, H100 logs, and
selection receipt; only then use the 4090 to establish direct-model fit and
browser latency.

`configs/llama_omni2_4090.yaml` remains a superseded preparation record because
the pinned LLaMA-Omni2 release lacks the required trainer. Never use
`train-s2s --allow-experimental` as the direct-model result: it reconstructs
the input and its checkpoints are deliberately runtime-ineligible.

## 4. Live official evaluation without raw-audio retention

For each consented participant, use a unique session ID and stable
pseudonymous speaker ID. The current service launches the selected cascade; do
not label the session as direct-Moshi evidence:

```bash
.venv/bin/python -m thesis_s2s.cli serve --study --retention features \
  --host 127.0.0.1 --port 8765 \
  --session-id MOS01 --speaker-id P01 --age-bin under_60
```

For an elderly participant use `--age-bin 60plus`. The browser must generate
`playback_started` for every persisted turn and `playback_stopped_ack` for every
labelled interruption; server-only timings are diagnostic. A missing or invalid
acknowledgement remains an explicit failure. Use Piper or a validated neural
talker, never the formant fallback, for rated sessions.
Before recruitment, confirm `/health` reports `validated_cascade_ready=true`.
The server binds every session to the ASR backend, responder model/revision and
prompt profile, plus the Piper model SHA-256. Any missing identity, model/rules
fallback, ASR/responder error, or mixed system identity fails the official gate.

After all sessions:

```bash
.venv/bin/python -m thesis_s2s.cli export-recordings
.venv/bin/python -m thesis_s2s.cli train-bargein \
  --recorded-jsonl data/processed/manifests/recorded.jsonl
.venv/bin/python -m thesis_s2s.cli study-summary \
  --out results/eval/human_study.json
```

Feature mode writes no WAV. Metrics-only mode is available if acoustic aggregates are also prohibited, but it cannot supply feature-based detector training evidence.

Do not accept the summary merely because `official_ready` is present. Require
all of these machine-checked fields to pass: `official_e2e_eligible=true`, zero
`official_failures_or_timeouts`, maximum `official_t_first_audio_max_ms` ≤500,
zero `official_runtime_failure_rows`, complete and consistent `system_provenance`,
engineering `official_t_barge_in_p95_ms` ≤300, proposal
`official_t_barge_in_max_ms` ≤150, naturalness `mos_mean` ≥3.5, 5–10
participants with at least two aged 60+, and independently labelled detector
accuracy **and** interrupt F1 strictly above 0.80.

## 5. Report and freeze

Report p50, p95, and maximum first-audio and barge-in values. Apply the
conservative maximum-first-audio ≤500 ms rule, report the separate engineering
barge-in p95≤300 ms rule, and apply the proposal-level maximum-barge-in ≤150 ms
rule. Keep component and live-client results in separate tables and retain all
failures/timeouts in the denominator.

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
