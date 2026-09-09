# Detailed proposal alignment audit

**Audited source:** local ignored `Thesis Proposal Template.docx`, SHA-256
`658a75f01f0d25cb18e601f8f5d36438c0c3780666a0f95c8fe529a6d68f90b4`

**Audit date:** 2026-09-08

## Bottom line

The project is strongly aligned with the proposal's research direction, but it
does not yet satisfy the proposal as a strict acceptance contract. The
machine-readable audit deliberately reports three different quantities:

- **100% traceability:** all five primary proposal objectives map to explicit
  artifacts and evidence gates.
- **80% substantial implementation coverage:** data, direct-model adaptation,
  streaming transport, and barge-in work all have real implementations; the
  participant study has only prepared tooling.
- **0% strict acceptance closure (0/5):** none of the five proposal-level
  success criteria is fully proven under a conservative reading.

The 80% figure measures workstream coverage, not scientific success. It must
never be reported as “the proposal is 80% complete.” The strict figure is also
not a grade: it reflects conjunctive thresholds where one missing condition
keeps the whole criterion open.

Regenerate the bound receipt with:

```bash
.venv/bin/python scripts/audit_proposal_alignment.py
```

## Primary success criteria

| Proposal criterion | Current evidence | Status |
|---|---|---|
| At least 100 h annotated Persian conversation with turn/overlap labels | Frozen training derivative: 6,754 pairs / 108.584 exported stereo h. Recommended balanced v2: 6,551 pairs / 110.374 source-pair h / four channels, 54.0% maximum source share, zero missing files/reused spans/group leaks; labels remain automatic and listening QA is explicitly waived | Partially aligned; quantity, integrity, and v2 source-balance pass, annotation confidence does not |
| Fine-tune a direct S2S model; lower Persian loss and produce intelligible speech | Six Moshika experiment generations are preserved; v6.2 reduced in-sample text loss 32.20%, but every candidate passed 0/9 direct-runtime rows | Partially aligned; training exists, intelligible direct output fails |
| Streaming end-to-end latency at or below 500 ms on one target GPU | Streaming/WebSocket transport and client acknowledgement instrumentation exist; qualifying physical 12–24 GB/browser rows are zero | Implemented but unproven; threshold not passed |
| Full duplex with at least 80% detector F1 and interruption handling within 150 ms | Continuous capture/cancel/next-turn transport is regression-tested. The automatic-label real-audio proxy has 81.06% accuracy but 78.99% interrupt F1; independently labeled and physical timing evidence is absent | Implemented but threshold/evidence fail |
| MOS at least 3.5 from 5–10 Persian speakers, including at least two aged 60+ | Consent, session, retention, prompt, rating, and analysis tooling exists; complete ratings are zero | Not performed; criterion fails |

## Architecture and outcome alignment

The proposal's central product claim is a direct spoken-input to spoken-output
model with **no intermediate text**. The project's positive production
candidate is instead:

```text
NeMo Persian ASR -> Qwen3-4B prompt-v2 -> Mana Persian Piper
```

That cascade is a real, positive Persian prototype: it passed the frozen
40-row automatic semantic final test and the nine-row real-service mechanics
panel. It is nevertheless not the architecture promised by the proposal. The
direct Moshika branch is scientifically useful as a well-controlled negative
result, not a deployment-eligible substitute. Consequently the title and any
claim of having delivered a direct S2S LLM remain stronger than the evidence.

## Research-question alignment

| Proposal research question | What the project answers | Remaining gap |
|---|---|---|
| Adapt an English/Chinese S2S model to Persian cheaply | Reproducible Moshi/Moshika LoRA recipes, six experiment generations, memory probes, frozen selection gates, and an honest negative runtime result | No intelligible deployable Persian direct checkpoint |
| Architectural changes for full duplex under 500 ms | Dual-buffer/continuous-input transport, lightweight acoustic detector, cancellation, pre-roll carryover, acknowledgements, and state-isolation tests | No physical browser proof and no qualifying <=150/500 ms result |
| Trade-off among latency, size, and quality | Controlled ASR, responder, and TTS candidate comparisons; H100 memory/runtime diagnostics; error and confidence reporting | No unified physical-target Pareto study and no human speech-quality axis |
| Elderly Persian users' perception | The full consent/metrics-only study apparatus and age-stratified schema are implemented | No complete participant, elderly, MOS, responsiveness, or qualitative data |

## Methodology consistency and justified deviations

| Proposal method | Implemented method | Assessment |
|---|---|---|
| Existing corpora plus optional self-recording | Public YouTube-derived Persian conversations; no self-recorded corpus | Consistent: self-recording was optional, and restricted media is not released |
| MFA-style forced alignment | Caption reconstruction, diarization, timestamp alignment, non-reuse and session-group audits | Method differs but serves the same segmentation objective; label quality remains automatic |
| Noise, simulated interruptions, variable SNR and room responses | Synthetic/noise fixtures and real automatic interaction candidates support detector engineering | Partial; not a fully human-labeled augmented training/evaluation corpus |
| Mini-Omni2/Qwen2.5-Omni example backbone | Moshika 7B with the official Moshi-Finetune trainer | Justified candidate substitution; Moshi has a native duplex objective and public adaptation path |
| Tokenizer adaptation plus CTC, LM and mel losses | Official Moshi audio/text-token losses with LoRA, selective embeddings and text-input dropout | Material methodological deviation; reproducible and documented, but the proposal should not claim the originally listed composite loss was used |
| Single 12–24 GB GPU for adaptation/evaluation | H100 training/profiling; physical RTX 4090 protocol prepared but not run | Training compute is acceptable engineering support, but proposal-level target-hardware evidence is missing |
| Loopback latency, timed interruptions and human study | Protocols, telemetry, fixtures and study forms are complete | Execution is missing for the physical and human stages |
| Well-commented reproducible open source | Apache-2.0 project code, pinned upstreams, privacy checks, tests, audits and GitHub release history | Strongly aligned; third-party/data licenses are correctly separated |

## Expected-outcome alignment

- **Working low-latency direct full-duplex Persian S2S:** not achieved. A
  working cascade and full-duplex control path exist, while the direct branch
  remains negative and latency is not officially measured.
- **Persian conversational dataset:** achieved in internal, automatically
  annotated form and at the required quantity. The post-training balanced v2
  also passes the four-source/55%-ceiling audit, but is not retroactively
  attributed to completed training. Raw data is not eligible for public release
  and labels are not human-verified.
- **Consumer-GPU latency and detector measurements:** not achieved. H100 and
  automatic proxy evidence cannot be relabeled as RTX 4090/independent evidence.
- **Elderly-user qualitative findings:** not achieved.
- **Open-source code:** achieved for repository-owned work under Apache-2.0;
  models, runtimes, template assets and restricted media retain separate terms.

## Proposal ambiguities that affect scoring

The source is visibly a template: student, university and supervisor fields are
unfilled, signatures are blank, and its final note says an undergraduate thesis
may prioritize either latency or full duplex. No filled/signed scope choice is
present. Treating it as binding is therefore conservative.

It also contains two material internal metric conflicts:

1. The problem statement requires **p90** latency <=500 ms, while the objective
   table requires **average** latency <=500 ms. The timing endpoint is not
   defined. Existing project documents conservatively report p50, p95 and max
   and do not claim the gate.
2. The detector objective writes “accuracy >=80% (F1 score),” although accuracy
   and F1 are different. The current proxy demonstrates why this matters:
   accuracy is 81.06% while interrupt F1 is 78.99%. Neither is official because
   labels are automatic.

## Defensible conclusion

The implementation is more rigorous and reproducible than the proposal asks
for in software, experiment governance, privacy, licensing, data leakage
prevention, negative-result reporting, and component comparison. It is less
aligned in the proposal's decisive scientific outcomes: direct architecture,
target-hardware latency, independently labeled interruption performance, and
human/elderly evaluation. The safe thesis claim is therefore:

> A reproducible Persian full-duplex cascade prototype with positive automatic
> semantic evidence, plus a comprehensive but unsuccessful direct-S2S
> adaptation study—not a completed low-latency direct Persian S2S system.
