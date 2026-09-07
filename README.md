# Persian full-duplex speech-to-speech (BSc thesis)

An evidence-first Persian speech prototype that keeps the microphone active during assistant playback and can stop the real browser audio source on a classical barge-in decision.

## Current status

| Requirement | Honest status |
|---|---|
| Final automatic semantic test | **Passed every predeclared gate on 40/40 group-disjoint test rows:** NeMo Persian ASR → exact Qwen3-4B-Instruct-2507 prompt-v2 → Piper. Across 240/240 valid blinded judge calls, relevance was **3.150 vs 1.425** for the frozen 0.5B baseline, coherence was **3.325**, relevance gain **+1.725**, paired win rate **67.5%**, and 87.5% of rows scored relevance ≥2 |
| Working spoken mechanics | **Passed 9/9** predeclared validation rows with audited user channel 1 → NeMo → exact Qwen2.5-0.5B baseline → Piper, 0/9 fallbacks, Persian-script replies, and non-silent audio. This historical panel anchors mechanics and round-trip intelligibility |
| Synthesized-speech intelligibility proxy | **Measured validly on the same 9/9 outputs:** exact transcript/reply hashes reproduced, zero ASR failures, NeMo round-trip micro CER **7.14%** (47/658 character edits) and micro WER **30.41%** (52/171 word edits). This automatic single-voice proxy is not a human pronunciation/naturalness result |
| Full-duplex control | Continuous browser PCM16 stream, rolling energy/F0/MFCC detector, server stop event, client stop acknowledgement, and tested transport-level carry-over of the captured interruption into the next turn. Physical-browser traces remain pending |
| Direct speech LLM | Experimental, not deployed. V1–v5 remain negative under their frozen runtime gates. The train-only v6.2 scheduled-text-dropout diagnostic achieved a **32.20%** text-loss reduction from step 50 to its best later checkpoint, but steps 50/100/150/200 each passed **0/9** direct-runtime rows. This proves objective learning capacity only; no direct adapter is deployment-eligible and no v2–v6.2 final test was opened. |
| 100–200 h conversation corpus | Full inventory: **775.887 h / 1,442 long episodes**. The production selection is **219.946 candidate h / 309 episodes** across four channels; 1,129 windows contain **207.154 automatically classified multi-speaker h**, **242.445 aligned staging h**, and **6,754 non-reused response pairs / 123.796 source-pair h**. The immutable Piper derivative is **108.584 measured stereo h**, inside the formal band |
| Conversational interruption supervision | Raw speaker boundaries recover **770 conservative candidates** (717 interruption-like, 53 backchannel-like) from the 6,754 pairs. The preserved deterministic 24-row sheet was sampled from the earlier 712-candidate pool and covers all four channels / **168.3 seconds** of excerpt audio. They remain automatic candidates; **zero human-verified direct interruptions** are claimed under the waiver |
| Barge-in >80% | Recorded-audio automatic-label proxy: 81.06% accuracy / 78.99% interrupt F1 on 132 events from 22 held-out sessions, but the session CI is 74.44–87.18% and human-verified labels are zero. Independent-label official evidence remains pending |
| ≤500 ms and 12–24 GB official test | Pending live browser measurements on a physical 12–24 GB GPU |
| Human study | Incomplete; 5–10 participants and at least two aged 60+ are still required. Raw WAV retention is optional |
| Project source license | **Apache-2.0 selected.** It covers repository-owned code/documentation only; third-party software, weights, datasets, adapters, restricted media, and generated derivatives retain their separate terms |

The headline positive result is
`results/eval/qwen4b_responder_v2_final_test_proxy.json`. It was opened only
after the frozen seven-row development eligibility stage passed. All 240
expected judge calls were valid and every frozen automatic gate passed. This
is a same-family Qwen LLM-as-judge proxy, not human evaluation or an independent
benchmark; it does not establish factuality, safety, naturalness, physical
4090 fit, or browser latency.

The earlier mechanics result is
`results/eval/cascade_real_service_validation_panel.json`. Machine-generated
privacy-safe descriptive analysis of that exact report is in
`results/eval/cascade_validation_descriptive_analysis.json`; it reads and
emits aggregate statistics and hashes, not transcript or reply plaintext. The
separately predeclared automatic synthesized-speech proxy is in
`results/eval/cascade_intelligibility_proxy.json`; its NeMo round-trip CER/WER
does not replace the separate automatic semantic test or human listening.
Corpus evidence is in `results/diarized_episode_audit_combined_authorized.json`,
`results/conversation_yield_estimate_combined.json`, and
`results/corpus_audit.json`; study evidence is in
`results/eval/human_study.json`. The final semantic panel establishes a
positive predeclared automatic test result; human naturalness, independent
benchmarking, and official browser latency remain outside its claim boundary.
Neither synthetic latency nor an H100 memory cap is accepted as official
end-to-end evidence.

Two additional local Apache-2.0 checkpoints, Qwen3.5-0.8B and Qwen3.5-4B,
were also verified to initialize and produce Persian in bounded synthetic
text-only smoke tests. They are useful future low-memory and upgrade candidates,
respectively, but neither was allowed to replace the frozen Qwen3-4B responder:
they have not passed a newly predeclared group-disjoint semantic evaluation,
and the already-open final panel cannot be reused for model selection. See
`results/hardware/qwen35_local_smoke.json`.

For the deadline handoff, `docs/THESIS_REPORTING_FA.md` provides copy-ready
Persian text for the abstract, methods, exact result tables, discussion,
limitations, and conclusion without promoting any pending evidence gate.

## Architecture

```text
continuous 16 kHz microphone ─┬─> rolling energy/F0/MFCC barge-in ─> stop browser source
                              │                                      └─> retain mic pre-roll as next turn
                              └─> NeMo ASR ─> Qwen3-4B prompt-v2/rules ─> Piper ─> full PCM reply
```

The bounded v3 experiment restored the pinned trainer's rank-128,
embedding-frozen LoRA default while keeping v2 data, context, optimizer, seed,
and output gates unchanged. V4 then changed exactly one factor by training only
`text_emb.weight` and `depformer_text_emb.weight` alongside the same rank-128
LoRA, with all 23 audio embeddings frozen. V5 then reduced only the first
semantic-codebook loss multiplier from 100 to 10. V3–v5 completed steps
100–500 and failed the unchanged nine-row official-runtime eligibility rule.
Selection returned null and the final-test firewall remained closed for every
version. V6/v6.1 then isolated train-only Persian capacity and deterministic
decoding; v6.2 added scheduled text-input dropout. Its positive in-sample
learning signal did not transfer to reliable direct generation: all four
candidate checkpoints passed 0/9 rows.

The frozen cascade v4 development panel passed 9/9, but a subsequent audit
found that it had fed assistant channel 0 to ASR. Before any validation-row
execution, `docs/CASCADE_VALIDATION_PROTOCOL.md` corrected the input to audited
user channel 1 and froze nine floor-spaced validation rows. The unchanged
NeMo/Qwen/Piper chain then passed 9/9 with zero fallbacks and zero split leaks.
That validation result—not the earlier train panel—is the working-system
evidence.

The subsequent frozen recovery compared that historical 0.5B responder with
Qwen3-4B-Instruct-2507. Prompt-v1 passed six of seven validation gates but
missed coherence by 0.1, so it remained negative. Prompt-v2 passed a separately
frozen seven-row development eligibility stage and unlocked the 40-row test
exactly once. The final test passed all validity and engineering gates with no
candidate retries or failed judge calls. Automatically aligned next-speaker
references scored poorly, explaining why responder LoRA reduced validation
loss by 31.18% yet degraded semantic proxy scores: those turns are useful
speech-timing material, but weak instruction-answer targets without curation.

The project server remains deliberately cascade-only until a Persian adapter passes every gate. The v1 adapter loads in Kyutai's pinned official server but fails autoregressive Persian output, so it cannot activate the direct path. Checkpoint metadata alone can never activate either that failed adapter or the legacy reconstruction artifact.

Post-finalization storage cleanup retains 13 representative v1–v5 adapters:
v1 steps 500/1000/2000/4000/8000, v2 steps 400/2000, and v3/v4/v5 steps
400/500. Their hashes match the committed experiment certificates. All
candidate losses, runtime outputs, configurations, hashes, and negative
verdicts remain tracked, but tensors for other non-promoted candidates were
deliberately removed; recreating them requires rerunning the frozen training
recipe. The chained receipts record 41.01 GiB reclaimed; see
`results/hardware/storage_cleanup_20260831.json`.
Four exact v6.2 diagnostic checkpoints currently remain and postdate that
cleanup receipt.

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
# Offline deployment: link the verified model tree without copying 8 GB.
ln -s /path/to/Qwen3-4B-Instruct-2507 models/qwen3-4b-instruct-2507
```

The provenance tests require full Git history (CI uses `fetch-depth: 0`). A
`--depth 1` checkout intentionally cannot validate historical launcher hashes;
use a normal clone for the final reproducibility audit.

Use either a protected preconfigured rclone remote via `THESIS_RCLONE_REMOTE` or ephemeral `S3_*` environment variables (see `.env.example`). Credentials are never copied from rclone configuration, placed in the repository, or passed as command arguments.
If a key has ever been stored in a plaintext cheatsheet or exposed in output,
rotate it and replace the document with environment-variable placeholders.

## Core commands

The general commands below are current. Versioned Moshi post-training commands
are also preserved as protocol/reconstruction commands: evaluating every v2–v5
candidate now requires first recreating the removed negative intermediate
tensors by rerunning the frozen experiment. They must not be used to reopen a
final test or revise a finalized selection.

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
.venv/bin/python -m thesis_s2s.cli evidence-status
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

# Frozen positive working-system validation (write-once result already exists):
.venv/bin/python scripts/evaluate_cascade_validation.py

# Frozen automatic intelligibility proxy (write-once result already exists):
.venv/bin/python scripts/evaluate_cascade_intelligibility.py

# Completed train-only v6.2 capacity diagnostic:
.venv-moshi/bin/python scripts/reevaluate_moshi_v6_overfit.py \
  --training-config configs/moshi_h100_v6_text_dropout.yaml \
  --run-dir checkpoints/moshi_v6_text_dropout \
  --metrics-out checkpoints/moshi_v6_text_dropout/metrics.reeval.jsonl \
  --report-out results/moshi_v6_text_dropout_reevaluation.json \
  --experiment-label v6_text_dropout
.venv-moshi/bin/python scripts/evaluate_moshi_v6_overfit.py --text-dropout-followup

# Historical v4 negative-result reproduction: rank-128 LoRA plus exactly two
# text embeddings. The completed run selected no adapter and never accessed its
# frozen final test; see docs/MOSHI_V4_RESULT.md.
.venv/bin/python scripts/preflight_moshi_v4.py
.venv/bin/python scripts/record_moshi_profile_probe.py --run-dir checkpoints/moshi_h100_100s_v4_text_embed_probe --probe-config configs/moshi_h100_v4_probe.yaml --full-config configs/moshi_h100_v4.yaml --embedding-policy configs/moshi_v4_embedding_policy.json --out results/hardware/moshi_h100_v4_profile_probe.json
.venv/bin/python scripts/run_moshi_v4_posttraining.py --training-invocation-id <systemd-invocation-id> --launch-commit <training-launch-commit>
.venv-moshi/bin/python scripts/record_moshi_v4_training_run.py --launch-commit <training-launch-commit> --service-unit <systemd-unit> --invocation-id <systemd-invocation-id>
# Only after an eligible schema-6 selection and certificate are committed:
.venv/bin/python scripts/run_moshi_v4_final.py

# Historical v5 negative-result reproduction: change only the semantic
# codebook multiplier from 100 to 10. No adapter was selected and the final test
# was never accessed; see docs/MOSHI_V5_RESULT.md.
.venv/bin/python scripts/preflight_moshi_v5.py
.venv/bin/python scripts/run_moshi_v5_posttraining.py --training-invocation-id <systemd-invocation-id> --launch-commit <training-launch-commit>
.venv-moshi/bin/python scripts/record_moshi_v5_training_run.py --launch-commit <training-launch-commit> --service-unit <systemd-unit> --invocation-id <systemd-invocation-id>
# Only after an eligible schema-7 selection and certificate are committed:
.venv/bin/python scripts/run_moshi_v5_final.py

# Recorded-audio automatic-label detector proxy (not official ground truth):
.venv/bin/python scripts/evaluate_recorded_bargein_proxy.py

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
`docs/MOSHI_V4_SELECTION_PROTOCOL.md`, `docs/MOSHI_V4_RESULT.md`,
`docs/MOSHI_V5_SELECTION_PROTOCOL.md`, `docs/MOSHI_V5_RESULT.md`,
`docs/MOSHI_V6_TEXT_DROPOUT_PROTOCOL.md`,
`docs/CASCADE_REAL_SERVICE_PROTOCOL_V4.md`,
`docs/CASCADE_VALIDATION_PROTOCOL.md`,
`docs/RECORDED_BARGEIN_PROXY_PROTOCOL.md`,
`docs/RIGHTS_REVIEW_FA.md`, `docs/SUPERVISOR_DECISIONS.md`, and
`docs/REVIEW.md`.

The ordered evidence and delivery path to a fully complete project is tracked
in `docs/PROJECT_10_OF_10_CHECKLIST.md`.

## License

Repository-owned source code and documentation are licensed under the Apache
License 2.0; see `LICENSE`. This grant excludes third-party components, model
and voice weights, datasets, trained artifacts, restricted media/captions, and
participant material. Their separate terms and release boundaries are recorded
in `THIRD_PARTY_NOTICES.md` and `docs/DATA_PROVENANCE.md`.
