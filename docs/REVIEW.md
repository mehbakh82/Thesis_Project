# Comprehensive project review (updated 2026-08-27)

## Verdict

The project has a strong thesis problem, useful infrastructure, and unusually good data-engineering effort, but the earlier implementation overstated two central claims: the trained artifact was not an end-to-end speech LLM, and the browser was not full duplex. The current revision corrects those claims, provides a functional modular cascade plus genuine continuous-microphone interruption control, constructs an auditable conversational training set, and adds a pinned official Moshi/Moshika LoRA path for genuine response-audio adaptation.

Current engineering/research readiness: **8.8/10**. Earlier audited state: **about 4/10**. A defensible 10/10 cannot be produced entirely in code because the remaining points require a deployment-eligible Persian direct model, reviewer listening, human participants, and live measurements on the forthcoming physical 4090.

The student has now explicitly waived both conversation listening reviews for
the time-constrained limited training run. This improves auditability, not the
score: strict readiness remains 8.8/10 because the reviews were designed and
preserved but not performed.

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

The original ASR ingest remains a substantial reusable result, but it is no longer presented as conversational supervision. A separate conversation pipeline now prepares, hashes, aligns, diarizes, noise-stratifies, and audits whole episodes while preserving episode-isolated splits.

The final authorized staging set contains 309 episodes and 1,129 windows (244.733 h). The deterministic reserve selector targets the binding final corpus measure rather than padding audio: 219.946 h of candidate audio, 207.154 h automatically classified as multi-speaker, and 242.445 h with aligned captions. The estimator and production builder both yield 6,754 adjacent-turn pairs / 123.796 source-pair h across Digiato, Mehran Rowshan Persian, Tabaghe16, and Zoomit. The file-backed waiver audit verifies 162 sessions, 407 speaker IDs, zero missing files, zero reused spans, zero split leakage, and all 6,754 pairs authorized; strict human-QA coverage remains false. Inputs, decisions, pairs, waiver, and audit are content-hashed.

The project's supervisor approved internal research training on the crawled public YouTube material. The authorization report therefore admits all 1,129 windows for internal training while correctly leaving redistribution disabled; internal-use approval is not represented as an open-content license.

The remaining corpus limitations are explicit:

- the 40-row, channel/outcome/noise-stratified listening sheet remains 0/40 and
  is waived—not completed—for the active limited run;
- 5,411 pairs retain the conservative `overlap_unattributed` label;
  all 770 raw-boundary interaction candidates remain automatic pseudo-labels,
  with zero human-verified interruption evidence under the waiver;
- speaker diarization estimates adjacent turns; it does not prove semantic user/assistant roles or response quality;
- YouTube media and generated training manifests remain internal-only;
- the synthetic harmonic fixture is only a plumbing smoke test, never speech evidence.

### 3. Model

The legacy `persian_omni2.pt` contains projector/ASR/mel-head weights only. Qwen was loaded but never used in `forward()`; its LoRA was not saved. The text target was formerly Python's process-randomized `hash(text)`, and the mel target reconstructed the input utterance rather than an assistant response. Piper then bypassed the learned model for audible output.

Corrections:

- stable BLAKE2 targets replace Python hash;
- the old training is explicitly named an experimental reconstruction ablation;
- it requires `--allow-experimental`;
- checkpoints declare `runtime_ready: false`;
- legacy/untyped checkpoints fail closed;
- serving is unconditionally cascade-only for the legacy artifact;
- the genuine direct path now uses the pinned official Moshi runtime and LoRA trainer, a fail-closed stereo exporter, a production-audit-clean pinned local client, deterministic single-voice Persian assistant targets, immutable upstream metadata, and exact model-file verification. All three base blobs pass their pinned sizes and SHA-256 hashes. The waiver-bound export is complete at 6,754 pairs / 108.584 final stereo h; an independent audit verifies every artifact/channel/source interval with zero failures and reproduces 24/24 sampled assistant renders exactly. The exact full-shape H100 probe completed model/Mimi/data/loss/backward/fused-AdamW at loss 3.808453 and 22.707 GB peak, then wrote a real 967 MiB/699-tensor adapter through the CPU-offloaded checkpoint path. It was launch evidence only; subsequent v1 and v2 scientific runs completed but both failed the official-runtime generation gates described below.

The later scientific v1 run completed all 8,000 steps and objective held-out
controls, but official-server validation exposed a decisive generation failure:
step 500 remained English and steps 1,000–8,000 became near-silent with no text.
The matched base control ruled out transport failure, so v1 is correctly
runtime-ineligible. A separately frozen v2 restores the official 100-second
context, adds validation speech/text/Persian gates, and reserves a fresh
14-session final test before training; its 22.797 GB one-step probe passes.

The clean v2 retry subsequently completed 2,000 steps and improved complete-
scope validation loss monotonically from 2.312407 to 1.734574. Its first panel
was invalid because eight of nine five-second inputs truncated active user
turns. After an input-only complete-prompt correction was frozen, all 20
checkpoints were reevaluated through the official runtime with unchanged gates.
No checkpoint passed 9/9 rows; step 400 was best at 2/9 and later checkpoints
again tended toward silence. V2 therefore failed closed, no adapter was
promoted, and its fresh final test was not accessed.

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
| Requirement alignment and claim discipline | 9 | 22 | 25 |
| Architecture | 6 | 19 | 20 |
| Implementation correctness | 7 | 18 | 20 |
| Data engineering and provenance | 9 | 14 | 15 |
| Evaluation quality | 2 | 6 | 10 |
| Reproducibility, tests, security | 3 | 9 | 10 |
| **Total** | **36/100** | **88/100** | **100/100** |

## Irreducible path to 10/10

1. Complete the 40-row window review and the 24-row interaction-candidate review. The latter uses existing internal-authorized audio and requires no recording; use a supplement or scope amendment only if its measured precision is unacceptable.
2. Continue controlled Moshi adaptation experiments on the H100 until one
   passes the unchanged validation-only Persian speech/text gates; v1 and v2
   are preserved negative results and must not be presented as deployment-ready.
3. Recruit 5–10 Persian speakers, including at least two aged 60+, and complete all ratings. If recruitment is formally waived, record the supervisor-approved alternative and narrow the claims accordingly.
4. Run the live browser protocol on the physical 4090 and export client timing; the H100 is the correct training machine, while the 4090 is the target deployment/evaluation machine.
5. Evaluate the detector on speaker/session-held-out real conversational audio and report confidence intervals.
6. Obtain human naturalness/satisfaction results for the final speech path.
7. Confirm the repository release license with the author and preserve model/environment checksums plus the restricted-data provenance manifest. This is separate from permission to train internally on the source corpus.

Until these evidence-producing steps are completed, claiming 10/10 would reduce rather than improve the thesis quality.

The documented student waiver enables a narrower automatic-label training
experiment but does not close item 1 and is not represented as supervisor
approval. Thesis methods and limitations must use the disclosure in
`docs/QA_WAIVER.md`.
