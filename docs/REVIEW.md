# Comprehensive project review (2026-08-22)

## Verdict

The project has a strong thesis problem, useful infrastructure, and unusually good data-engineering effort, but the earlier implementation overstated the two central claims: the trained artifact was not an end-to-end speech LLM, and the browser was not full duplex. The current revision corrects those claims and provides a functional modular cascade plus genuine continuous-microphone interruption control.

Current engineering/research readiness: **7.9/10**. Earlier audited state: **about 4/10**. A defensible 10/10 cannot be produced entirely in code because the definition requires external conversational evidence, a genuine S2S adaptation, 5–10 human participants, and live measurements on the forthcoming physical 4090.

## Stage-by-stage assessment

### 1. Definition and planning

Good choices:

- The latency, interruption, hardware, corpus, and elderly-participant gates were extracted explicitly.
- Reuse of the local NeMo Persian service and existing YouTube inventory is efficient.
- The plan eventually corrected the mistaken NeMo-teacher detour and restored CSV captions.

Problems:

- LLaMA-Omni2 was treated as if its training stack were open and locally reproduced. The official release is primarily inference-oriented.
- PersonaPlex memory/language assumptions were optimistic; its published model is English-oriented and evaluated on much larger hardware.
- Component timing and synthetic detector results were repeatedly allowed to stand in for end-to-end evidence.

### 2. Corpus

The 197.613 h ingest is a substantial and reusable result. Episode hashing gives group-isolated train/val/test splits, and the full audit found no duplicate utterance IDs, schema errors, or episode leakage.

Caption alignment is reasonably supported, but not proven: the independent-ASR proxy over 110,024 Digiato rows / 101.091 h has mean token-sequence similarity 0.7243 and 6.13% below 0.4; a seeded live-ASR sample of 20 Zoomit clips has mean 0.7735 and none below 0.4. Because ASR disagreement is not ground truth, manual stratified listening remains required before model training claims.

It is not, however, a 197.6 h conversational duplex corpus:

- all 215,684 rows have `interrupt_label=none`;
- no rows contain overlap annotations;
- no rows contain user-to-assistant response supervision;
- the sources are largely long-form technology monologues;
- 24 exact texts occur across split boundaries (a warning, not episode leakage);
- YouTube media remains internal-only.

The synthetic 20 h artifact is tiled harmonic audio. It is appropriate for a plumbing smoke test, not scientific evidence of speech robustness.

### 3. Model

The legacy `persian_omni2.pt` contains projector/ASR/mel-head weights only. Qwen was loaded but never used in `forward()`; its LoRA was not saved. The text target was formerly Python's process-randomized `hash(text)`, and the mel target reconstructed the input utterance rather than an assistant response. Piper then bypassed the learned model for audible output.

Corrections:

- stable BLAKE2 targets replace Python hash;
- training is explicitly named an experimental reconstruction ablation;
- it requires `--allow-experimental`;
- checkpoints declare `runtime_ready: false`;
- legacy/untyped checkpoints fail closed;
- serving is unconditionally cascade-only; a future direct model needs a separately implemented/tested loader as well as a validated artifact.

The working system is now honestly modular: NeMo ASR → locally cached Qwen2.5-0.5B (rules if unavailable) → Piper/formant TTS.

### 4. Barge-in

The energy/F0/MFCC feature direction matches the definition. The new controller maintains a rolling 450 ms window and requires consecutive positive hops. It records a conservative acoustic-onset bound when no explicit onset is supplied.

The prior 1.00 accuracy is only harmonic synthetic evidence. Recorded evaluation is now split by speaker/session so a participant cannot leak across train and test. The >80% thesis claim remains pending until consented held-out recordings exist.

### 5. Runtime

The former UI recorded only between button presses, uploaded audio after stopping, sent only 250 ms of reply, and did not stop the actual browser source.

The current protocol:

- streams PCM continuously while collecting or playing;
- resamples the browser's real capture rate to mono 16 kHz;
- uses echo cancellation while preserving raw dynamics for detection;
- sends complete reply PCM;
- stops all live `AudioBufferSourceNode` instances on a server detector event;
- acknowledges the actual client stop;
- reports client-observed first-audio and stop timing;
- never substitutes synthetic audio for an empty microphone turn.

### 6. Study, privacy, and security

Sessions now require an explicit consent checkbox. Audio, lossy-feature, and metrics-only retention modes are explicit; the default study mode stores no WAV. IDs are path-safe, existing sessions are not truncated, identity conflicts fail, labels/prompts are validated, and ratings use bounded 1–5 fields. S3 credentials are no longer parsed from shell aliases or placed in process arguments.

`study-summary` reports rating and detector confidence intervals and refuses readiness until all participant, elderly, complete-rating, client-timing, physical-GPU, and real held-out >80% detector gates pass.

### 7. Evaluation and reproducibility

Automated latency JSON is now permanently labeled a server/synthetic component proxy and cannot become official merely because a suitable GPU is present. The official path is consented browser telemetry.

Added:

- standards-compliant YAML parsing;
- manifest integrity/coverage audit;
- strict metric input validation;
- isolated tests that do not retrain into shared results;
- Ruff and CI configuration;
- corrected dependency declarations;
- secret-free environment template;
- a natural-conversation response-pair builder with session-level splitting and fail-closed audit;
- immutable upstream commit/license metadata;
- physical-GPU preflight and release-snapshot hashing;
- literal maximum latency reporting alongside p50/p95.

## Scoring

| Dimension | Before | Current | Maximum |
|---|---:|---:|---:|
| Requirement alignment and claim discipline | 9 | 20 | 25 |
| Architecture | 6 | 17 | 20 |
| Implementation correctness | 7 | 17 | 20 |
| Data engineering and provenance | 9 | 12 | 15 |
| Evaluation quality | 2 | 5 | 10 |
| Reproducibility, tests, security | 3 | 8 | 10 |
| **Total** | **36/100** | **79/100** | **100/100** |

## Irreducible path to 10/10

1. Build the definition's 100–200 h genuinely conversational Persian set with response pairs, noise, overlap, and interruption labels. New raw-audio recording is optional unless the supervisor explicitly requires it; natural conversation with a verified license or documented internal-research authorization is the preferred no-recording route.
2. Recruit 5–10 Persian speakers, including at least two aged 60+, and complete all ratings.
3. Run the live browser protocol on a physical 12–24 GB GPU and export client timing.
4. Train or adapt a genuine speech-conditioned causal language model with supervised assistant speech tokens; validate Persian output and save every runtime component.
5. Evaluate the detector on speaker/session-held-out real data and report confidence intervals.
6. Obtain human naturalness/satisfaction results for Piper or a validated Persian neural speech decoder.
7. Confirm the release license with the author and archive an immutable version-control commit, environment/model checksums, and restricted-data provenance manifest.

Until these evidence-producing steps are completed, claiming 10/10 would reduce rather than improve the thesis quality.
