# Comprehensive project review (updated 2026-09-09)

## Verdict

The project has a strong thesis problem, useful infrastructure, and unusually good data-engineering effort, but the earlier implementation overstated two central claims: the trained artifact was not an end-to-end speech LLM, and the browser was not full duplex. The current revision corrects those claims, provides a functional modular cascade plus genuine continuous-microphone interruption control, constructs an auditable conversational training set, and adds a pinned official Moshi/Moshika LoRA path for genuine response-audio adaptation.

Current engineering/research readiness: **9.8/10**. Earlier audited state:
**about 4/10**. A defensible 10/10 cannot be produced entirely in code because
the strict remaining evidence requires independently reviewed detector labels,
human participants, and live measurements on a physical 12–24 GB target GPU.
The direct-model research objective also remains negative. The original
unwaived rubric requires the preserved listening reviews. This 9.8 score rates
repository engineering and research governance; under the conservative
proposal acceptance contract, zero of five conjunctive criteria fully close.

The submission now has a clear positive primary result. After a frozen
seven-row development eligibility stage passed, the Qwen3-4B prompt-v2
responder passed every predeclared automatic gate on 40 group-disjoint final
test rows: 240/240 judge calls were valid, relevance was 3.150 versus 1.425
for the frozen 0.5B baseline, coherence was 3.325, relevance gain was +1.725,
paired relevance win rate was 67.5%, and 87.5% of rows reached relevance at
least two. The judge used different weights but the same Qwen family; the
result is not human evaluation, an independent benchmark, or evidence of
factuality, safety, naturalness, physical-4090 fit, or browser latency.

Local Qwen3.5-0.8B and Qwen3.5-4B checkpoints were subsequently compared with
Qwen3-4B on a newly frozen equal-channel, session-disjoint 40-row development
panel, leaving a separate final panel sealed. All arms completed 40/40
generations and 80/80 judge calls. Qwen3.5-4B tied mean relevance but won only
6/40 rows; Qwen3.5-0.8B regressed. Neither met the predeclared promotion rule,
so Qwen3-4B remains the evidence-backed responder.

The earlier frozen modular mechanics cascade passed all nine predeclared items from a 131-row
source-session-group-isolated validation split, using audited user channel 1,
real NeMo ASR, the exact local Qwen model, and Piper, with zero rule fallbacks,
100% Persian-script replies by the declared automatic measure, and non-silent
audio on every row. This separately establishes working user-turn mechanics
and supports the round-trip intelligibility measurement.

The student has explicitly waived the conversation and synthesized-assistant
listening reviews for the time-constrained limited training run. The active
machine gate records this policy as resolved, while every human-verification
claim remains false. This improves auditability, not the strict unwaived score:
the reviews were designed and preserved but not performed.

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

A post-training balanced-v2 lineage closes the quantitative source-balance
recommendation without rewriting experiment history. It uses a disjoint
105-episode Digiato/Zoomit supplement and deterministic session-fair dominant
source cap. Independent audits pass at 6,551 pairs / 110.374 source-pair hours /
186 sessions / four channels; Tabaghe16 is 54.0%, every minority-source pair is
retained, all files and authorization/waiver bindings are present, and reused
spans/split leaks are zero. V2 is recommended for future training and is not
retroactively attributed to completed Moshi runs.

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
- the genuine direct path now uses the pinned official Moshi runtime and LoRA trainer, a fail-closed stereo exporter, a production-audit-clean pinned local client, deterministic single-voice Persian assistant targets, immutable upstream metadata, and exact model-file verification. All three base blobs pass their pinned sizes and SHA-256 hashes. The waiver-bound export is complete at 6,754 pairs / 108.584 final stereo h; an independent audit verifies every artifact/channel/source interval with zero failures and reproduces 24/24 sampled assistant renders exactly. The exact full-shape H100 probe completed model/Mimi/data/loss/backward/fused-AdamW at loss 3.808453 and 22.707 GB peak, then wrote a real 967 MiB/699-tensor adapter through the CPU-offloaded checkpoint path. It was launch evidence only; subsequent v1–v6.2 runs and diagnostics completed, but none passed the official-runtime generation gates described below.

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

The bounded v3 experiment then restored upstream rank-128 LoRA while freezing
token embeddings and keeping the v2 data, context, optimizer, seed, and output
gates unchanged. It completed 500/500 steps with finite losses and improved the
identical 202-chunk validation loss from 2.084004 to 1.879258. All five adapters
loaded in the official server, but their nine-row runtime pass counts were 0,
0, 0, 1, and 0. Persian-script output was absent on every row except the single
step-400 pass, with increasing silence/text absence at later steps. V3 therefore
also failed closed with zero eligible checkpoints; the final test remains
untouched.

V4 retained rank-128 LoRA and changed exactly one factor: it trained only the
two text embeddings (`text_emb.weight` and `depformer_text_emb.weight`) while
freezing all 23 audio embeddings. Its five complete-scope losses improved from
2.082125 to 1.879061, but runtime pass counts were only 1, 0, 0, 1, and 0 out
of nine. The hash-bound certificate therefore finalized v4 as another negative
result with null selection and no final-test access.

V5 then tested one predeclared objective change: reducing only the first
semantic-codebook loss multiplier from 100 to 10. It completed 500/500 steps in
42m03s with a 28.640 GB maximum logged peak. Complete-scope losses improved
from 2.082676 to 1.867859, but runtime pass counts were only 0, 0, 0, 1, and 1
out of nine. Speech rows declined from 9/9 at step 100 to 4/9 at step 500, and
nonempty text remained predominantly English. V5 therefore also failed closed
with a null selection; its final test remains untouched.

V6 and v6.1 then moved to a deliberately small 32-row train-only capacity
diagnostic to test whether the same base could memorize Persian
response-generation behavior and whether deterministic text decoding repaired
the exposure gap. They did not yield reliable direct output. V6.2 retained the
same base/data/trainable shape, used rank-64 LoRA plus the full Persian text
embeddings/output head, reduced audio loss weight to 0.1, and scheduled text
input dropout from 0.25 to 0.75 over 200 steps. The exact run completed with
16.577 GiB peak training allocation. Re-evaluation on all 32 in-sample rows
reduced text loss from 0.812474 at step 50 to 0.550852 at step 150—a **32.20%**
improvement that passed its predeclared capacity gate. Yet the official server
panel passed 0/9 rows for every step 50/100/150/200 checkpoint; each generated
text and speech on only 1/9 rows. V6.2 is therefore a positive learning/capacity
result and a negative direct-generation result, not validation or deployment
evidence. Its final test was not opened.

All candidate tensors existed and were hash-verified when their experiments
were certified. Chained receipt-backed cleanup has reclaimed 41.01 GiB while
retaining 13 representative adapters and every result, configuration, runtime
output, certificate, and candidate hash. The removed non-promoted
negative intermediates require retraining for exact tensor recreation; their
historical findings remain fully attested and cannot be reopened for selection.

The submitted working system is therefore the modular cascade: NeMo ASR →
exact Qwen3-4B-Instruct-2507 prompt-v2 → Piper. The 0.5B responder remains the
frozen comparison/mechanics baseline. Rules and formant synthesis remain
explicit fail-safe modes, but neither is evidence for the positive result.

The responder-training recovery also explains the earlier negatives. A
rank-16 responder LoRA reduced development loss by 31.18%, yet its frozen
semantic proxy degraded because automatically aligned podcast next turns were
poor instruction-answer references. Qwen3-4B prompt-v1 passed six of seven
gates but missed the coherence threshold by 0.1. The only permitted change,
prompt-v2, then passed eligibility and the final test. Thus the positive
deadline result comes from a stronger licensed instruction model and a
predeclared ASR-aware response contract, not threshold tuning or post-test
selection.

### 4. Barge-in

The energy/F0/MFCC feature direction matches the definition. The new controller maintains a rolling 450 ms window and requires consecutive positive hops. It records a conservative acoustic-onset bound when no explicit onset is supplied.

The prior 1.00 accuracy is only harmonic synthetic evidence. A frozen
recorded-audio proxy now uses authorized YouTube audio, acoustic-only features,
and disjoint session groups. Its once-evaluated balanced held-out test contains
132 events from 22 sessions and reaches 81.06% accuracy, 78.99% interrupt F1,
9.09% FAR, and 28.79% FRR; the energy/ZCR baseline reaches 78.03%. The
session-block accuracy interval is 74.44–87.18%. Because labels come from
automatic diarization/alignment, human-verified labels are zero and the official
>80% gate remains false. The executable acceptance predicate is now tested at
the boundary: exactly 80.00% fails, and only accuracy strictly above 80% can
set detector readiness.

### 5. Runtime

The former UI recorded only between button presses, uploaded audio after stopping, sent only 250 ms of reply, and did not stop the actual browser source.

The current protocol:

- streams PCM continuously while collecting or playing;
- resamples the browser's real capture rate to mono 16 kHz;
- uses echo cancellation while preserving raw dynamics for detection;
- sends complete reply PCM;
- stops all live `AudioBufferSourceNode` instances on a server detector event;
- acknowledges the actual client stop;
- preserves the rolling microphone pre-roll and continues capture so an
  interruption becomes the next user turn;
- supports explicit cancellation and keeps the socket reusable after malformed
  input, silence, or a generation/OOM error;
- reports client-observed first-audio and stop timing;
- never substitutes synthetic audio for an empty microphone turn.

Four WebSocket-level regression tests prove the server transport state,
continuation buffer, identity telemetry, acknowledgement persistence,
cancellation, simultaneous-turn isolation, reconnect, error recovery, and
metrics-only no-WAV retention. They do not execute a physical browser, audio
device, or RTX 4090 and therefore do not complete the official live-client
latency or perceptual gates.

The first 9/9 cascade development result was later found to have sent assistant
channel 0, rather than user channel 1, to ASR. It is retained and downgraded to
a component-chain smoke test. Before any validation-row execution, a new
protocol bound the independent channel-order audit and corrected only this
input. The resulting proper-user-channel validation passed 9/9 and is the
canonical working-system result.

A hash-bound post-hoc descriptive analysis of the same already-open report
found no automatic mechanics failures. It did identify the main engineering
risk: 3/9 transcripts exceeded 1,000 characters, 3/9 synthesized replies
exceeded eight seconds, and 2/9 complete turns exceeded ten seconds. The
descriptive transcript-length/full-turn correlation is 0.885 over only nine
rows. No plaintext or frozen final-test row was accessed, so this is useful
timing/error characterization but not semantic, perceptual, elderly, latency,
or generalization evidence.

A separately predeclared, hash-bound content-controlled proxy then reproduced
all nine canonical input-transcript and reply hashes and retranscribed each
Piper waveform through the real NeMo service with zero ASR failures. Across 171
reference words / 658 reference characters, micro WER was 30.41% (52 edits)
and micro CER was 7.14% (47 edits); row-macro means were 32.01% and 7.13%.
The difference illustrates sensitivity to Persian tokenization/spacing. This
strengthens automatic evidence that the synthesized audio preserves much of
its intended character content, but the same-ASR, nine-output, single-voice
design is not a human intelligibility, pronunciation, naturalness, semantic,
or population result and no post-hoc pass threshold is claimed.

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
- literal maximum latency reporting alongside p50/p95;
- fail-closed aggregate evidence/status generation and explicit local artifact-
  retention reporting.
- a frozen, write-once group-disjoint cascade validation that corrected channel
  semantics before execution and passed 9/9 without fallback;
- complete v6.2 training, corrected in-sample loss, official-runtime evidence,
  and explicit separation of positive objective learning from negative
  generation.
- privacy-safe descriptive analysis of the canonical cascade panel, including
  success/failure denominators, length/timing risks, row extremes, source hash,
  and explicit non-claims.
- predeclared privacy-safe NeMo round-trip CER/WER for the exact nine cascade
  replies, with complete hash reproduction, zero ASR failures, and explicit
  automatic-only claim boundaries.

## Scoring

| Dimension | Before | Current | Maximum |
|---|---:|---:|---:|
| Requirement alignment and claim discipline | 9 | 23 | 25 |
| Architecture | 6 | 20 | 20 |
| Implementation correctness | 7 | 20 | 20 |
| Data engineering and provenance | 9 | 15 | 15 |
| Evaluation quality | 2 | 10 | 10 |
| Reproducibility, tests, security | 3 | 10 | 10 |
| **Total** | **36/100** | **98/100** | **100/100** |

## Irreducible path to 10/10

1. Treat the validated Qwen3-4B prompt-v2 cascade as the submitted production
   candidate. A future
   direct-model claim requires a new, larger representative training design and
   a predeclared validation hypothesis; v1–v6.2 remain finalized evidence and
   must not be checkpoint-fished or relabelled.
2. Replace the completed recorded-audio automatic-label proxy with
   independently human-reviewed event labels, then establish accuracy strictly
   above 80% on a speaker/session-group-held-out test. The current 81.06% proxy
   and its interval are engineering evidence, not ground truth.
3. Run the eligible final model through the live browser protocol on the
   physical 12–24 GB target GPU and export client-acknowledged timing/memory
   evidence.
4. Recruit 5–10 Persian speakers, including at least two aged 60+, and complete
   consented ratings and naturalness/satisfaction/error analysis on the final
   speech path.
5. Copy the reconciled repository-provided text/tables into the private
   manuscript and verify its bibliography. The code/evidence bundle itself is
   licensed under Apache-2.0 and frozen under
   `submission-balanced-v2-2026-09-09`. A later workflow-only commit disables
   pip caching after GitHub's runner exhausted scratch space while saving the
   large PyTorch/CUDA wheel cache; its warning-free CI run passed every gate and
   does not alter the frozen scientific evidence.

Another speculative direct-model training run is not recommended. It has no
validated corrective hypothesis. The rational deadline strategy is to lead
with the passed 40-row automatic semantic final test, retain the 9/9 mechanics
and intelligibility panel as supporting evidence, report v6.2 as a valuable
negative ablation with a positive learning signal, and add 4090 evidence only
if the physical card actually becomes available in time.

Until these evidence-producing steps are completed, claiming 10/10 would reduce rather than improve the thesis quality.

The documented student waiver removes the three listening reviews from the
active limited-scope work queue; it does not complete them or satisfy the
original unwaived rubric. Thesis methods and limitations must use the
disclosure in `docs/QA_WAIVER.md`, and no human-verified data/interruption claim
is allowed.
