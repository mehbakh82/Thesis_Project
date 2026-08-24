# Code map

## Deployable path

- `src/thesis_s2s/runtime/duplex.py` — FastAPI UI, continuous 16 kHz microphone stream, WebSocket protocol, browser playback acknowledgements, and study endpoints.
- `src/thesis_s2s/runtime/cascade.py` — bounded NeMo HTTP ASR → local Qwen/rules response → Piper/formant TTS. This is the current working speech-to-speech path.
- `src/thesis_s2s/runtime/tts.py` — Piper synthesis with a formant fallback for development only.
- `src/thesis_s2s/runtime/session_log.py` — consent-gated, path-safe session WAV/JSONL storage and evidence-readiness summary.
- `src/thesis_s2s/bargein/` — energy/F0/MFCC features, GBDT training, energy baseline, and rolling real-time playback controller.

## Direct model training path

- `estimate-conversation-yield` and `select-conversation-reserve` — non-mutating pair-yield estimation plus deterministic minimum whole-episode reserve selection under an explicit candidate-hour cap; neither claims training readiness.
- `src/thesis_s2s/data/noise.py` — deterministic speech/nonspeech RMS estimate for real archived background-noise conditions; labels remain automatic until listening QA.
- `src/thesis_s2s/data/moshi.py` — fail-closed conversion from authorized non-reused Persian response pairs to the official Moshi stereo dialogue schema, using a deterministic pinned Persian assistant voice for the primary run and source responses only as an ablation.
- `configs/moshi_h100.yaml` — single-H100 LoRA profile for text and Mimi assistant-speech-token losses; `configs/moshi_h100_profile_probe.yaml` measures the exact training shape for one non-scientific step before launch.
- `third_party/UPSTREAMS.lock.json` — immutable Moshi runtime/trainer revisions and license boundaries.
- `scripts/build_moshi_client.py` plus
  `third_party/overlays/moshi-client/` — exact-source/hash verification,
  production-audit-clean static client build, and per-file bundle attestation.
- `docs/MOSHI_H100_RUNBOOK.md` — environment, data, H100 training, runtime, and 4090 handoff procedure.
- `src/thesis_s2s/repro.py` — fail-closed readiness report separating H100 hardware, trainer stack, two listening-QA stages, final Moshi export, transient full-profile headroom, and physical 4090 evaluation.

## Experimental model path

- `src/thesis_s2s/model/llama_omni2.py` — encoder/reconstruction ablation. It is **not** a speech-conditioned conversational LLM and its artifacts are not runtime-ready.
- `train_s2s()` in `src/thesis_s2s/model/llama_omni2.py` — split-aware experimental training, gated by `--allow-experimental`.
- `checkpoint_runtime_status()` in `src/thesis_s2s/model/llama_omni2.py` — fail-closed artifact validation for health/evidence reporting. Serving remains cascade-only until a direct runtime is separately implemented and tested.

## Data and evaluation

- `src/thesis_s2s/data/ingest.py` and `data/s3_inventory.py` — CSV inventory and per-episode rclone streaming through either protected named remotes or environment-only credentials; no credentials enter argv.
- `src/thesis_s2s/data/prepare_youtube.py` — Tabaghe16 caption/audio join and 16 kHz WAV preparation.
- `src/thesis_s2s/data/batch_reasr.py` — optional diagnostic NeMo/Whisper re-ASR; never overwrites canonical captions.
- `src/thesis_s2s/data/verbatim.py` — Persian normalization while retaining source-caption provenance.
- `src/thesis_s2s/data/filter_corpus.py` — quality filters and synthetic plumbing data.
- `src/thesis_s2s/data/diarize.py`, `manual_qa.py`, `interaction_qa.py`, and `conversation.py` — collision-free primary/reserve reports, transcript-aligned speakers, stratified window QA, conservative raw-boundary interaction candidates, fail-closed pair review, overlap-preserving response-pair building, and verified-only interruption audits.
- `src/thesis_s2s/data/audit.py` — manifest integrity, split leakage, conversational-supervision coverage, and caption/independent-ASR alignment-risk audits.
- `src/thesis_s2s/data/factory.py` — data pipeline orchestration.
- `src/thesis_s2s/bakeoff/codecs.py` — codec/component probes, not a comparative full-model benchmark.
- `src/thesis_s2s/eval/bench.py` — component proxy measurements and evidence metadata. Official end-to-end latency comes only from live browser telemetry on eligible physical hardware.
- `src/thesis_s2s/metrics.py` — validated latency, detector, and hardware-gate calculations.

See `docs/REVIEW.md` for the audit verdict and the remaining evidence required by the thesis definition.
