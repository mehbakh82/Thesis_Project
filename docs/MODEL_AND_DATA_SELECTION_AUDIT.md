# Model and data selection audit

**Audit date:** 2026-09-08
**Evidence class:** requirements-constrained engineering audit; not a claim of
global model optimality

This document closes an ambiguity in the earlier project language. The chosen
components are the strongest options verified inside this repository, but the
original component survey did not run a controlled Persian comparison of every
current model. The machine-readable source catalog is
`configs/model_selection_audit.yaml`; regenerate its fail-closed receipt with:

```bash
thesis-s2s audit-model-selection
thesis-s2s audit-conversation-balance
```

Unknown evidence never passes. Official upstream documentation can establish a
released capability, but only a project measurement can establish Persian
quality on the project distribution.

## Decision that remains valid

The submitted production candidate remains:

```text
NeMo Persian ASR -> Qwen3-4B-Instruct-2507 prompt-v2 -> Mana Persian Piper
                         + continuous-input barge-in controller
```

It is the only path with positive, frozen Persian system evidence: 40 valid
group-disjoint final rows, all seven automatic semantic gates passed, relevance
3.150 versus 1.425 for the frozen baseline, coherence 3.325, and 67.5% paired
wins. Newer models must not be selected by reusing that opened final panel.

The direct Moshika branch remains an experimental negative result. Moshika is a
reproducible engineering baseline because the official Moshi-Finetune project
provides a stereo conversational objective and loadable LoRA adapters. It is
not a Persian winner: v6.2 reduced in-sample text loss by 32.20%, while all four
checkpoints passed 0/9 frozen direct-runtime rows.

## Current candidate audit

| Track | Current selection | Current selection eligible? | Why no global-best claim |
|---|---|---:|---|
| Direct S2S | Moshika 7B engineering baseline | No | Failed Persian runtime output; no same-panel comparison |
| ASR | Fine-tuned Persian NeMo service | Yes | Controlled and post-hoc screens retained NeMo; Rade/Omnilingual remain unevaluated and the references are automatic captions |
| Responder | Qwen3-4B prompt-v2 | Yes | Controlled three-arm development comparison retained it; evidence is automatic-proxy and catalog-bounded, not a global claim |
| TTS | Mana Persian Piper | Yes | Controlled automatic comparison retained Piper; human listening remains absent |

The current direct shortlist is Moshika, PersonaPlex, Qwen3-Omni,
MiniCPM-o 4.5, Covo-Audio-Chat-FD, BayLing-Duplex, and DuplexOmni. None has
project-verified Persian speech output plus a public adaptation path and a
documented 24 GB deployment path. The current catalog links only official
repositories/model cards and retains `unknown` wherever those sources do not
establish a requirement.

The candidate refresh inspected official repositories/model cards available on
2026-09-08. In addition to the evaluated Qwen3-ASR and MMS releases, it covers
Shenava-Koochik, Rade-ASR-CTC-3B-fa, and MOSS-TTS-Nano-Persian. This is a
dated, requirements-constrained catalog—not a permanent or exhaustive claim
over every unpublished, gated, or future checkpoint.

The highest-value component challengers are:

1. **ASR:** the controlled post-release comparison is complete. Both
   Qwen3-ASR challengers were substantially worse than fine-tuned NeMo on the
   frozen Digiato/Zoomit development panel. A later post-hoc Shenava screen
   was far faster but also less accurate, so NeMo is retained. Rade-ASR's exact
   Torch/fairseq2/Omnilingual runtime imported successfully on CUDA, but a
   deadline-bounded attempt could not complete its 6.16 GB checkpoint transfer;
   therefore no inference or quality result exists. Even a favorable post-hoc
   screen could not fairly promote it from the open development panel.
2. **Responder:** the controlled post-release comparison is complete. Local
   Qwen3.5-4B tied the incumbent's mean relevance and improved mean coherence
   by 0.075, but its 15% row win rate and zero relevance gain failed the frozen
   promotion thresholds. Qwen3.5-0.8B regressed. Retain Qwen3-4B prompt-v2.
3. **TTS:** the controlled automatic comparison is complete. Meta MMS-TTS
   Persian was mechanically valid but materially worse than Mana-Piper, so
   Piper is retained. Released Qwen3-TTS support still excludes Persian.
   MOSS-TTS-Nano-Persian requires clean reference speech unavailable to this
   project and documents a roughly five-second practical utterance limit, so a
   fair like-for-like deployment comparison was not run.
4. **Direct S2S:** retain Moshika only as the reproducible negative baseline.
   Do not start another full run until a candidate passes a 32-row Persian
   memorization and autoregressive-output gate.

## Completed responder comparison

`scripts/evaluate_responder_candidates_v1.py` froze a new equal-channel panel
before inference. Development used 40 train-partition rows from eight source
sessions (10 rows per channel); the separately hashed 40-row final panel uses
eight different sessions and remains unopened. All three arms completed 40/40
generations and 80/80 judge calls:

| Arm | Relevance | Coherence | Generation p50 | Outcome |
|---|---:|---:|---:|---|
| Qwen3-4B prompt-v2 | 3.475 | 3.650 | 1,481 ms | Retained incumbent |
| Qwen3.5-4B | 3.475 | 3.725 | 1,617 ms | Not promoted: +0.000 relevance, 6/40 row wins |
| Qwen3.5-0.8B | 3.125 | 2.400 | 1,805 ms | Not promoted: relevance/coherence regression |

The predeclared challenger rule required at least +0.25 mean relevance,
nonnegative coherence, and a 55% row-level relevance win rate. Because neither
challenger passed it, opening the new final panel would add test exposure
without changing the decision and is prohibited by the protocol. These scores
use a deterministic same-family automatic judge; they do not establish human
preference, factuality, safety, or global model optimality. See
`docs/RESPONDER_CANDIDATE_COMPARISON_V1.md` and the hash-only receipts under
`results/eval/responder_candidates_v1_*`.

## Completed ASR comparison

`scripts/evaluate_asr_candidates_v1.py` froze a leakage-conscious panel before
inference. Development used 40 rows from eight Digiato/Zoomit sessions (20 per
channel; 272.465 seconds). These sources were not listed in the documented
approximately 250-hour Tabaghe16 NeMo fine-tuning set. A separate 40-row,
eight-session final panel (264.190 seconds) remains unopened.

| Arm | Micro CER | Micro WER | RTF p50 | Outcome |
|---|---:|---:|---:|---|
| Fine-tuned NeMo | 0.230119 | 0.371593 | 0.109569 | Retained incumbent |
| Qwen3-ASR-1.7B | 0.383788 | 0.651363 | 0.233951 | Not promoted |
| Qwen3-ASR-0.6B | 0.492509 | 0.807747 | 0.195577 | Not promoted |

For Qwen3-ASR-1.7B, challenger-minus-NeMo CER had a paired session-bootstrap
95% interval of [0.115957, 0.211482]; for 0.6B it was
[0.220173, 0.303799]. Both are entirely on the worse side of zero. NeMo's
Digiato/Zoomit CER was 0.272185/0.185420; 1.7B's was
0.421327/0.343899; 0.6B's was 0.492916/0.492076.

An initial attempt incorrectly omitted one valid empty NeMo hypothesis from
the aggregate denominator. It was preserved as a failed receipt, the evaluator
was corrected so empty output receives full-deletion error counts, a regression
test was added, and the same exact panels and thresholds were re-frozen under
new code/protocol hashes before the reported run. This comparison uses public
automatic captions rather than human-clean transcripts, and it establishes a
catalog-bounded choice rather than global ASR optimality. See
`docs/ASR_CANDIDATE_COMPARISON_V1.md` and the receipts under
`results/eval/asr_candidates_v1_*`.

### Post-hoc Shenava supplemental screen

Shenava-Koochik was discovered after the ASR v1 candidates and thresholds were
frozen. Retroactively inserting it into that experiment would misstate the
chronology, so `scripts/evaluate_asr_shenava_supplemental_v1.py` reused only
the already-open development panel and was frozen as an explicitly post-hoc
screen. It did not access the sealed final panel and could not directly promote
the challenger.

| Arm | Micro CER | Micro WER | RTF p50 | Outcome |
|---|---:|---:|---:|---|
| Fine-tuned NeMo rerun | 0.230119 | 0.371593 | 0.095592 | Retained incumbent |
| Shenava-Koochik sherpa-ONNX | 0.241644 | 0.403156 | 0.016045 | Did not advance |

Both arms completed all 40 rows. Shenava-minus-NeMo absolute CER reduction was
-0.011525 and WER reduction was -0.031563; the paired eight-session bootstrap
interval for its CER difference was [-0.016063, 0.044917]. Thus the challenger
failed the predeclared material-gain, nonworse-WER, and confidence gates. The
result establishes excellent CPU speed but no accuracy improvement on this
automatic-caption panel. Possible overlap between Shenava's reported
VisualEars training source and the project sources is unknown. See
`docs/ASR_SHENAVA_SUPPLEMENTAL_V1.md` and the hash/count-only receipts under
`results/eval/asr_shenava_supplemental_v1_*`.

## Completed TTS comparison

`scripts/evaluate_tts_candidates_v1.py` froze 40 project-representative
response texts from eight train-partition sessions, balanced 10 per source and
disjoint from both responder-candidate panels. Both arms synthesized all 40
texts and completed every NeMo round-trip call:

| Arm | Micro CER | Micro WER | Render RTF p50 | Outcome |
|---|---:|---:|---:|---|
| Mana Persian Piper | 0.174401 | 0.342520 | 0.033745 | Retained incumbent |
| Meta MMS-TTS Persian | 0.250892 | 0.494094 | 0.122488 | Not promoted |

MMS-minus-Piper CER had a paired session-bootstrap 95% interval of
[0.029315, 0.130514], entirely favoring Piper. The challenger therefore failed
the frozen minimum 0.02 CER-reduction and nonworse-WER gates; the separate
40-text final panel remains sealed. The MMS checkpoint was pinned to revision
`8818d36618d125a0b40b5d2b2713a852877e9b68`; its CC-BY-NC-4.0 weights are not
redistributed and do not inherit this repository's Apache-2.0 license.

This establishes only automatic round-trip intelligibility and full-render
latency on one ASR. It does not establish human naturalness, pronunciation,
speaker preference, or true first-audio streaming latency. See
`docs/TTS_CANDIDATE_COMPARISON_V1.md` and the hash-only receipts under
`results/eval/tts_candidates_v1_*`.

## Controlled comparison protocol for remaining or future tracks

A future promotion must use a new, frozen, session-disjoint development and
test set. It must not reuse either the opened 40-row Qwen3-4B final panel or the
new sealed responder-candidate final panel.

### Stage 0: documentary elimination

Require an acceptable use/license boundary, public inference, exact model
revision, and a feasible local runtime. For direct S2S additionally require
native listen-while-speaking behavior and a public adaptation method. Unknown
fields fail closed.

### Stage 1: compatibility and hardware

Run each surviving candidate with identical Persian prompts/audio. Record model
and tokenizer hashes, runtime revision, dtype/quantization, peak whole-process
VRAM, load failures, timeouts, Persian-script output, non-silent audio, and
streaming behavior. A 24 GB claim requires a physical 12--24 GB device; an H100
memory cap is diagnostic only.

### Stage 2: controlled development selection

- ASR: identical natural user audio and reference captions; report CER/WER,
  failures, real-time factor, first partial result, and noise/source strata.
- Responder: identical transcripts and prompt contract; blinded order, the same
  independent judge/human rubric, relevance, coherence, factuality/safety
  errors, response length, failures, and latency.
- TTS: identical response text; round-trip CER/WER plus human pronunciation,
  naturalness, intelligibility, first-audio latency, real-time factor, and
  failures.
- Direct S2S: identical stereo training subset, optimizer budget, validation
  rule, runtime panel, Full-Duplex-Bench-compatible turn-taking measures, and
  physical-target fit.

Only development data may choose a model or threshold. Freeze the exact winner
before opening the new final test once.

### Stage 3: final test

Report every row and failure in the denominator, paired confidence intervals,
per-source results, and physical-target latency. A global-best claim is still
too broad; the justified claim is “best among the predeclared candidates under
this Persian protocol and hardware constraint.”

## Exact data balance result

The frozen corpus used by the completed Moshi experiments has 6,754 pairs,
162 sessions, and 123.795553 source-pair hours:

| Source | Pairs | Sessions | Hours | Hour share |
|---|---:|---:|---:|---:|
| Tabaghe16 | 3,881 | 77 | 92.782454 | 74.9481% |
| Mehran Rowshan Persian | 1,302 | 51 | 17.995783 | 14.5367% |
| Digiato | 1,104 | 21 | 7.945021 | 6.4179% |
| Zoomit | 467 | 13 | 5.072295 | 4.0973% |

Integrity remains strong, but this historical version does not pass the 55%
maximum-share target. The previously recorded remediation arithmetic was a
fixed-size replacement of 24.694900 Tabaghe16 hours or a retain-all addition of
44.899818 non-Tabaghe hours.

That recommendation is now implemented as a separate balanced v2 rather than
rewriting training history. It uses a hash-bound local-audio supplement,
versioned maximum-duration interval selector, duplicate-safe pair merge, and
deterministic session-round-robin cap of the dominant source:

| Source | Pairs | Sessions | Hours | Hour share |
|---|---:|---:|---:|---:|
| Tabaghe16 | 2,126 | 77 | 59.601947 | 54.0000% |
| Mehran Rowshan Persian | 1,580 | 51 | 24.800360 | 22.4694% |
| Digiato | 2,091 | 37 | 17.921230 | 16.2368% |
| Zoomit | 754 | 21 | 8.050468 | 7.2938% |

The final v2 total is **6,551 pairs / 110.374005 source-pair hours / 186
sessions**. Every minority pair is retained; the largest source is exactly
54.0%, below the independent 55% gate. File-backed audit reports zero missing
files, reused source intervals, invalid split rows, or group leaks and confirms
all 6,551 rows are internally authorized and bound to the same QA waiver. This
closes the quantitative representativeness recommendation without adding the
Iman Khoraminezhad or Karnakon channels. V2 is the recommended future corpus;
the completed Moshi results remain correctly attributed to frozen v1.

Kooshiar remains unsuitable as the main source of adjacent conversational
responses because it is predominantly monologue. It can be used in a separate
ASR/acoustic auxiliary set, never relabeled as duplex response supervision.

## Remaining strict limitations

These cannot be manufactured by code or inferred from automatic proxies:

- preserved listening QA remains unperformed;
- no independently labeled detector evaluation exists;
- no physical RTX 4090/browser latency run exists;
- no human Persian naturalness/pronunciation study exists;
- no current direct model produces deployment-eligible Persian speech;
- controlled current-candidate comparison remains incomplete for direct S2S;
  responder, ASR, and automatic TTS comparisons are complete and retained
  Qwen3-4B, fine-tuned NeMo, and Mana-Piper, respectively.

The new audits improve the project by making those boundaries executable and
quantitative. They do not alter the frozen positive cascade result, the negative
Moshika result, or the existing LaTeX reports.
