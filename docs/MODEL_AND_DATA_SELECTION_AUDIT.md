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
| ASR | Fine-tuned Persian NeMo service | Yes | Controlled comparison retained NeMo; Omnilingual-ASR remains unevaluated and the references are automatic captions |
| Responder | Qwen3-4B prompt-v2 | Yes | Controlled three-arm development comparison retained it; evidence is automatic-proxy and catalog-bounded, not a global claim |
| TTS | Mana Persian Piper | Yes | No second Persian TTS passed the same intelligibility/latency panel |

The current direct shortlist is Moshika, PersonaPlex, Qwen3-Omni,
MiniCPM-o 4.5, Covo-Audio-Chat-FD, BayLing-Duplex, and DuplexOmni. None has
project-verified Persian speech output plus a public adaptation path and a
documented 24 GB deployment path. The current catalog links only official
repositories/model cards and retains `unknown` wherever those sources do not
establish a requirement.

The highest-value component challengers are:

1. **ASR:** the controlled post-release comparison is complete. Both
   Qwen3-ASR challengers were substantially worse than fine-tuned NeMo on the
   frozen Digiato/Zoomit development panel, so NeMo is retained.
2. **Responder:** the controlled post-release comparison is complete. Local
   Qwen3.5-4B tied the incumbent's mean relevance and improved mean coherence
   by 0.075, but its 15% row win rate and zero relevance gain failed the frozen
   promotion thresholds. Qwen3.5-0.8B regressed. Retain Qwen3-4B prompt-v2.
3. **TTS:** retain Mana-Piper unless a Persian-adapted challenger is evaluated.
   Released Qwen3-TTS language support does not include Persian.
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

## Controlled comparison protocol for remaining tracks

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

The existing natural-source manifest has 6,754 pairs, 162 sessions, and
123.795553 source-pair hours:

| Source | Pairs | Sessions | Hours | Hour share |
|---|---:|---:|---:|---:|
| Tabaghe16 | 3,881 | 77 | 92.782454 | 74.9481% |
| Mehran Rowshan Persian | 1,302 | 51 | 17.995783 | 14.5367% |
| Digiato | 1,104 | 21 | 7.945021 | 6.4179% |
| Zoomit | 467 | 13 | 5.072295 | 4.0973% |

Integrity remains strong: zero invalid rows in this audit, zero reused spans in
the existing conversation audit, and zero session-group split leaks. Source
representativeness does not pass the original 55% maximum-share target.

Two mathematically exact remediation choices are recorded:

- At the same 123.795553-hour total, replace at least **24.694900 hours** of
  Tabaghe16 pairs with other sources.
- If retaining every current Tabaghe16 pair, add at least **44.899818 hours**
  from non-Tabaghe sources, producing at least 168.695371 total hours.

The second plan remains within the thesis 100--200-hour band. Prefer new
podcasts such as Iman Khoraminezhad and Karnakon plus unused high-confidence
Mehran/Digiato/Zoomit sessions. Apply the same episode reconstruction,
diarization, alignment, authorization, non-reuse, and group-split audits. Do not
silently replace the frozen submission corpus: create a versioned v2 manifest
and rerun every dependent training/evaluation receipt.

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
- controlled current-candidate comparisons remain incomplete for TTS and
  direct S2S; the responder and ASR comparisons are complete and retained
  Qwen3-4B and fine-tuned NeMo, respectively.

The new audits improve the project by making those boundaries executable and
quantitative. They do not alter the frozen positive cascade result, the negative
Moshika result, or the existing LaTeX reports.
