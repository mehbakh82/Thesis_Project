# Responder comparison v3: result

Protocol: `docs/RESPONDER_V3_PROTOCOL.md` (frozen 2026-10-31, Amendments 1–5
and an exploratory addendum). Receipt: `results/eval/responder_v3_result.json`
(scores, timings, memory and SHA-256 hashes of the private panels, replies and
judgments; no reply or panel text).

**Outcome.**

- **24 GB profile: Gemma 4 12B QAT W4A16 (Google's official 4-bit build) on
  vLLM.** It has the highest test score among the configurations that fit and
  meet the latency gate (median first token 39 ms, 152 tokens/s, peak
  10.9 GB), and it beats both references under both judges: +1.20 (J1) and
  +1.75 (J2) points over Qwen3.5-4B, and +3.63 and +3.52 over the deployed
  Qwen3-4B, all with 95% intervals above zero.
- **12 GiB profile: no configuration both fits and meets the gate.** The vLLM
  builds of Gemma 4 12B need 10.9 GB, about 200–250 MiB over the 10,682 MiB
  limit, and the llama.cpp build that fits (Q5_K_M) takes 169–214 ms to its
  first token.
- **The first decision (2026-11-07) named Gemma 4 12B FP8.** Amendment 4 ran
  two builds that had been missed. Neither fits its profile, but rescoring
  every Gemma 4 12B build together moved the 24 GB finalist of that model from
  FP8 to QAT. The two builds score the same on the test panel (QAT minus FP8:
  −0.03 under J1, +0.12 under J2, both intervals across zero); QAT is smaller
  (10.9 against 15.4 GB) and as fast.
- **Sensitivity (Amendment 5):** five more spaced latency measurements of every
  finalist with the GPU load recorded, and a rescoring with more judge
  reasoning, change no gate result and no ranking that matters for the
  decision.
- **Deployment qualification of the 24 GB winner: not passed.** In the live
  stack it met memory (12,466 MiB), latency (mean 372 ms, p90 468 ms) and
  script (no Latin letter), but failed coherence after interruption: replies
  after an interruption were 0.37 points less relevant than the controls under
  both judges (the rule allows 0.25). It stays an opt-in preset, not the
  default. See "Deployment" below.
- **Post-hoc 12 GiB checks of Gemma 4 12B: neither passed.** vLLM with one
  sequence fits (11,768 MiB) but missed one reply and failed coherence under
  the Qwen3.8 judge. An exploratory diagnosis traced the slow llama.cpp first
  token to context checkpoints of Gemma 4's sliding-window cache (24.8 ms with
  `--swa-full`, at 10,088 MiB). That build then fit (11,700 MiB) but failed
  latency under shared-host load and coherence under both judges. In all three
  Gemma runs the model asks the user to repeat after a garbled follow-up
  (11–14 of 60 follow-ups, against 3 for Qwen3.5-4B).

## What was compared

- **Inventory:** 284 arms of 15 models (Qwen3.5 0.8B–35B-A3B, Qwen3.6 27B and
  35B-A3B, Qwen3.8-27B, Gemma 4 E2B–31B), every quantized build the two engines
  can run (vLLM `v0.30.0`, llama.cpp), official and community, plus the two
  references: Qwen3.5-4B bf16 with the elderly prompt (the best 4B) and the
  deployed Qwen3-4B-Instruct-2507 with prompt v2. 59 arms were excluded
  before any run: 32 whose weights fit no profile (25 over the file-size
  limit, 7 bf16 builds of the 26–35B models), 12 bitsandbytes builds that vLLM
  0.30 cannot load, and 15 PrismML ternary and binary builds that need a
  llama.cpp fork that was not built on this host.
- **Serving:** each configuration at the deployment requirement (context
  2,048, two concurrent sequences, 96 new tokens, greedy decoding, the
  Persian-only output constraint), with its measured peak GPU memory, the KV
  memory per sequence profiled at two contexts, and latency (median first
  token and decode rate) on the shared H100.
- **Profiles:** 12 GiB (responder at most 10,682 MiB, so the whole stack with
  the 1,606 MiB recognizer fits 12,288 MiB) and 24 GB (responder at most
  22,528 MiB). Latency gate: median first token at most 150 ms and decode at
  least 30 tokens/s.
- **Panels:** development 100 rows (50 elderly requests, 25 written by each
  judge family, and 50 natural user turns with two exchanges of history);
  test 252 rows (126 elderly requests, 63 per author, and 126 natural turns).
  Frozen with hashes before any generation; no row of the v2 panel is reused.
- **Judges:** J1 Claude Opus 5.5 (gateway) and J2 GPT-6 Astra (Codex,
  reasoning effort medium). Neither family has a candidate. Listwise scoring of
  up to 12 anonymized replies per call, 0–10 for relevance, coherence,
  appropriateness for an older listener and overall, with one planted reply
  written for another row in every call. A judge is unreliable if the planted
  reply scores 6 or more in more than 10% of calls.

## Judge reliability

| Stage (judge, panel) | Calls | Planted reply mean | Planted ≥ 6 |
|---|---:|---:|---:|
| Stage 1 model screen (J2, dev) | 200 | 1.41 | 5.5% |
| Stage 2 variant screen (J2, dev) | 400 | 1.52 | 3.8% |
| Stage 3 builds (J2, dev) | 695 | 2.10 | 8.6% |
| KV precision (J2, dev) | 100 | 1.80 | 6.0% |
| Finalists (J2, dev) | 100 | 1.70 | 7.0% |
| Test (J1, test) | 501 | 1.07 | 4.0% |
| Test (J2, test) | 501 | 1.62 | 6.0% |
| More reasoning (J1 thinking, J2 effort high; 60 test rows) | 60 + 60 | | 0% |

Every stage is under the 10% limit. Stage 3 is the closest (8.6%): its calls
compare builds of one model, whose replies are often near-identical, and the
planted reply is then the only clearly different one.

## Stage 1: models (best build of each model, J2, development panel)

| Model | Best build | Overall |
|---|---|---:|
| Gemma 4 12B | QAT W4A16 (vLLM) | 7.16 |
| Gemma 4 31B | Q4_K_M (llama.cpp) | 7.05 |
| Gemma 4 26B-A4B | Q4_K_M | 6.76 |
| Qwen3.8-27B | Q4_K_M | 6.76 |
| Qwen3.5-27B | Q4_K_M | 6.72 |
| Qwen3.5-35B-A3B | Q4_K_M | 6.44 |
| Gemma 4 E4B | Q4_K_M | 6.20 |
| Qwen3.5-9B | Q4_K_M | 6.20 |
| Qwen3.6-35B-A3B | Q4_K_M | 6.05 |
| Qwen3.6-27B | Q4_K_M | 5.59 |
| Qwen3.5-4B | Q4_K_M | 5.43 |
| Gemma 4 E2B | Q4_K_M | 5.37 |
| Qwen3.5-2B | Q4_K_M | 3.32 |
| Qwen3.5-0.8B | bf16 | 2.42 |

Gemma 4 leads at every size. The Qwen3.6 models often repeat or continue the
user's words on natural turns (their chat template was checked), so they rank
below Qwen3.5 of the same size.

## Stage 2: quantization methods (J2, development panel)

Every build of Qwen3.5-4B and Gemma 4 E4B, and the two builds of Qwen3.8-27B,
against the model's highest-precision build (paired mean overall difference,
5,000-draw row bootstrap; non-inferior if at least −0.3).

| Build | Qwen3.5-4B | Gemma 4 E4B |
|---|---|---|
| Q8_0 / Q6_K / Q5_K_M | +0.15 / +0.24 / +0.26 | +0.15 (Q8_0) |
| Q4_K_M | +0.12 | +0.06 |
| IQ4_XS, UD-Q4_K_XL | −0.04, −0.15 | |
| FP8, INT8 | −0.17, +0.07 | |
| MXFP4 | −0.21 | |
| Official QAT W4A16 | | +0.14 |
| AWQ, AutoRound, INT4 (compressed-tensors) | −0.51, −0.49, −0.52 (inferior) | |
| GPTQ INT4 | −0.92 (inferior) | |
| NVFP4 | −0.67 (inferior) | −0.51 (inferior) |
| Q3_K_M, UD-Q2_K_XL, Q2_K | −0.66, −0.43, −1.65 (inferior) | |

Qwen3.8-27B Q4_K_M: +0.14 against its official FP8 build.

- **8-bit and GGUF 4–6-bit builds lose nothing measurable.** Every K-quant
  from Q4_K_M up, IQ4_XS, FP8, INT8 and MXFP4 is non-inferior.
- **Most 4-bit vLLM formats lose 0.5–0.9 points.** AWQ, AutoRound,
  compressed-tensors INT4, GPTQ and NVFP4 are inferior on Qwen3.5-4B, and NVFP4
  also on Gemma 4 E4B. The official QAT build of Gemma 4 E4B is the exception.
- **3 bits or fewer are out.** Q3_K_M is inferior, so by Amendment 1 no
  build of 3 bits or fewer is tested further. This also means the PrismML
  builds, which were not run, would not have been tested beyond stage 2.
- Δq = −0.06 (no highest-precision build beat its Q4_K_M), so the contender
  margin is δ = 0.5.

## Stage 3 to 5: contenders, KV precision and finalists

- **Contenders (within δ of the best fitting stage 1 score):** Gemma 4 12B in
  the 12 GiB profile; Gemma 4 12B, 31B and 26B-A4B, Qwen3.8-27B and Qwen3.5-27B
  in the 24 GB profile.
- **Reduced KV precision (Amendment 3, non-inferior if ≥ −0.3 against the
  default KV):** Qwen3.5-4B bf16 with FP8 KV −0.14; Qwen3.5-4B Q4_K_M with
  q8_0 −0.07 and q4_0 +0.04; Gemma 4 31B Q5_K_M with q4_0 −0.11 (interval
  −0.30 to 0.08). All non-inferior.
- **vLLM 0.30 cannot start Gemma 4 with an FP8 KV cache:** with head size 512
  it falls back to an attention backend that rejects FP8. A first analysis
  scored these failed runs as empty replies; the harness now refuses absent
  reply sets, and those judgments are kept apart
  (`judgments_kvquality_j2.invalid_absent_replies.jsonl`, hashed in the
  receipt). This finding is what made Amendment 4 necessary.

- **Finalists (Amendment 3, after Amendment 4):** for each contender and
  profile, the fitting configuration with the best stage 3 score among those
  whose development run meets the gate (else the best fitting one).

| Profile | Model | Finalist | Stage 3 (J2) | First token (dev) | Gate |
|---|---|---|---:|---:|---|
| 12 GiB | Gemma 4 12B | Q5_K_M, llama.cpp | 7.41 | 214 ms | not met |
| 24 GB | Gemma 4 12B | QAT W4A16, vLLM | 7.13 | 55 ms | met |
| 24 GB | Gemma 4 31B | Q5_K_M, q4_0 KV, llama.cpp | 7.44 | 377 ms | not met |
| 24 GB | Gemma 4 26B-A4B | NVFP4, vLLM | 6.98 | 60 ms | met |
| 24 GB | Qwen3.8-27B | AWQ INT4, vLLM | 6.85 | 117 ms | met |
| 24 GB | Qwen3.5-27B | NVFP4, vLLM | 6.68 | 117 ms | met |

## Amendment 4: two builds that had been missed

The stage 3 builds were fixed on 2026-11-04, when Gemma 4 12B and 31B QAT
still counted as fitting with an FP8 KV cache. Once that proved impossible in
vLLM 0.30, Amendment 1 named the next vLLM build of each: NVFP4. Running them
(written up after the first decision had been read):

- **Gemma 4 12B NVFP4:** the first build (unsloth) fails at start with the same
  FP8 attention assertion; the fallback (RedHatAI) runs at a peak of
  10,912 MiB, so it misses the 12 GiB limit too. In the 24 GB profile it scores
  6.98 in stage 3, below QAT.
- **Gemma 4 31B NVFP4:** peak 23,920 MiB, over the 24 GB limit (22,528 MiB).
- **The rescoring changed one finalist.** Scoring the six Gemma 4 12B builds
  together again and averaging with the first stage 3 calls moved QAT (7.13)
  above FP8 (6.97); the Amendment 3 rule then makes QAT the model's 24 GB
  finalist. QAT got its test run and was scored on the test panel in one call
  per row with the eight configurations scored before.

## Test panel and decision (252 rows, both judges)

| Configuration | J1 overall | J2 overall | Mean | First token | Decode | Peak | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| **Gemma 4 12B QAT W4A16 (24 GB winner)** | 6.47 | 7.09 | **6.78** | 39 ms | 152/s | 10,914 MiB | met |
| Gemma 4 31B Q5_K_M, q4_0 KV | 6.86 | 6.89 | 6.87 | 152 ms | 55/s | 22,194 MiB | not met |
| Gemma 4 12B Q5_K_M (12 GiB finalist) | 6.70 | 7.18 | 6.94 | 179 ms | 117/s | 9,772 MiB | not met |
| Gemma 4 26B-A4B NVFP4 | 6.24 | 6.54 | 6.39 | 71 ms | 216/s | 17,282 MiB | met |
| Qwen3.8-27B AWQ INT4 | 6.16 | 6.41 | 6.28 | 113 ms | 100/s | 21,150 MiB | met |
| Qwen3.5-27B NVFP4 | 6.15 | 6.16 | 6.16 | 112 ms | 94/s | 20,656 MiB | met |
| Gemma 4 12B FP8 (first decision) | 6.49 | 6.97 | 6.73 | 51 ms | 139/s | 15,402 MiB | met |
| Qwen3.5-4B bf16 (best 4B) | 5.27 | 5.34 | 5.31 | 56 ms | 243/s | 9,634 MiB | met |
| Qwen3-4B, prompt v2 (deployed) | 2.84 | 3.58 | 3.21 | 35 ms | 253/s | 9,392 MiB | met |

Latency is the decision latency of Amendment 3 (the test run, or the median of
three runs where two differed by more than a factor of two: for QAT the
development run decoded at 70 tokens/s and the test run at 152).

- **The winner beats both references** under the stage 7 rule (overall gain at
  least 0.5 with the interval above zero, and no loss in relevance, coherence
  or appropriateness, under both judges):

| Gain of the winner over | Judge | Relevance | Coherence | Appropriateness | Overall (95% CI) |
|---|---|---:|---:|---:|---|
| Qwen3.5-4B | J1 | +0.48 | +1.45 | +1.57 | +1.20 (0.91 to 1.49) |
| Qwen3.5-4B | J2 | +1.19 | +1.14 | +1.98 | +1.75 (1.43 to 2.07) |
| Qwen3-4B (deployed) | J1 | +2.57 | +3.92 | +3.69 | +3.63 (3.33 to 3.93) |
| Qwen3-4B (deployed) | J2 | +2.37 | +3.51 | +3.48 | +3.52 (3.19 to 3.85) |

- **Gemma 4 is the strongest family again.** Every Gemma 4 finalist scores
  above every Qwen finalist under both judges, and every 9–31B finalist is at
  least 0.8 points above Qwen3.5-4B. This reverses the v2 size comparison
  (no larger model beat Qwen3.5-4B there), whose two judges each came from a
  candidate family and agreed on 52% of relevance scores.
- **The best-scoring configurations are slow ones.** Gemma 4 31B (6.87) and
  Gemma 4 12B Q5_K_M (6.94) score highest on the mean but run on llama.cpp and
  miss the gate; the winner's score is within 0.2 of both.
- **Natural turns separate the configurations most.** Under J1 every
  configuration scores lower on real caption turns than on written requests
  (by 0.05–1.48 points). The two Qwen 27B models lose about 1.5 points there
  under both judges, Gemma 4 12B the least. The winner has the best
  natural-turn score under J2 (7.21) and the second best under J1 (6.23,
  after Gemma 4 12B Q5_K_M at 6.27).

## Sensitivity: latency under GPU load (Amendment 5)

Five more latency-only measurements of every finalist and reference, at least
30 minutes apart, with the other processes' SM utilization and the host load
recorded once per second:

| Configuration | Measurements | First token, range | Median, all | Median, quiet | Share meeting gate |
|---|---:|---:|---:|---:|---:|
| Gemma 4 12B QAT (winner) | 8 | 38–55 ms | 40 ms | 39 ms | 100% |
| Gemma 4 26B-A4B NVFP4 | 7 | 49–71 ms | 53 ms | 52 ms | 100% |
| Qwen3.8-27B AWQ | 7 | 109–117 ms | 110 ms | 110 ms | 100% |
| Qwen3.5-27B NVFP4 | 7 | 98–166 ms | 112 ms | 106 ms | 86% |
| Gemma 4 31B Q5_K_M, q4_0 KV | 8 | 139–336 ms | 154 ms | 155 ms | 25% |
| Gemma 4 12B Q5_K_M | 7 | 169–214 ms | 179 ms | 175 ms | 0% |
| Qwen3.5-4B bf16 | 8 | 55–79 ms | 63 ms | 77 ms | 100% |
| Qwen3-4B (deployed) | 8 | 34–46 ms | 35 ms | 34 ms | 100% |

- **No gate result changes.** For every configuration, the median over all
  measurements and over the quiet ones (other processes at most 10% SM) agree
  with the gate result the decision used.
- **Load from other processes was rare** during the new measurements (other
  SM utilization 0% in 37 of 40). Two of the three exceptions (42–49%) halved
  the decode rate of the run they fell on (Qwen3.5-27B: 165 ms and
  43 tokens/s; Gemma 4 26B-A4B: 99 tokens/s instead of about 215), and the
  first-token time correlates only weakly with GPU load (rank correlation
  0.10 for vLLM). For llama.cpp it correlates more with the host's CPU load
  (0.40). The slowest early runs (Gemma 4 31B at 336 ms and 13 tokens/s, the
  4B references at about 115 tokens/s) predate the load record.
- **Gemma 4 31B misses the gate under every condition measured** but by a few
  milliseconds (149–157 ms in the five new runs); it is a borderline case, not
  an artefact of one slow run.
- **Variation without GPU load exists:** Qwen3.5-4B ranges from 55 to 79 ms with
  no other GPU activity, so part of the spread comes from the host (scheduling
  and the Persian-only constraint on the CPU), not from the GPU.

## Sensitivity: judge reasoning (Amendment 5)

The original test calls of the first 60 test rows (identical prompts), scored
again with more reasoning:

| | J1 with extended thinking | J2 with effort high |
|---|---:|---:|
| Rank correlation of the eight configuration means | 0.93 | 0.98 |
| Reply scores identical / within one point | 77% / 99.8% | 77% / 100% |
| Mean shift of the scores | −0.01 | −0.04 |
| Planted reply scored 6 or more | 0% | 0% |

The rankings move only among configurations within 0.1 of each other; the top
and bottom of both rankings and the gap to the references are unchanged. The
judges' reasoning setting does not drive the result. Candidate reasoning
(thinking) stays disabled, as frozen: every thinking token precedes the first
spoken word, so it cannot meet the latency requirement.

## Exploratory: can serving settings put Gemma 4 12B QAT in 12 GiB?

Outside the decision (protocol, exploratory addendum). Gemma 4 12B QAT W4A16
on vLLM, default KV, test panel:

| vLLM tuning | Peak | First token | Decode | Fits 12 GiB | Gate |
|---|---:|---:|---:|---|---|
| `--max-num-batched-tokens 512` | 11,050 MiB | 38 ms | 162 tok/s | no | met |
| `--enforce-eager` | 11,062 MiB | 91 ms | 26 tok/s | no | not met |
| both | 10,832 MiB | 94 ms | 25 tok/s | no | not met |

No tuning brings it under 10,682 MiB; the best is 150 MiB over, and only by
giving up CUDA graphs, which drops decoding below the gate. Its development
run had measured 10,924 MiB, so repeated peaks of one configuration vary by
about 100–150 MiB.

## Corrections to the harness, all before the numbers above were final

- **Absent replies:** an engine that crashed left no replies, and a first KV
  analysis scored them as empty. The harness now refuses absent reply sets;
  the invalid judgments are kept apart and hashed.
- **Missed builds:** Amendment 4.
- **Identical prompts:** when a new build's reply repeats another build's reply
  word for word, the judge prompt can be identical to one already judged, and
  its call had been skipped (5 stage 3 rows of Gemma 4 12B NVFP4, 3 test rows of
  Gemma 4 12B QAT). As the protocol's rule for identical replies requires, the
  judged scores now apply to every arm that gave that reply (no new judge
  calls; such records do not count as calls in the planted-reply statistics).
  This changed the winner's test mean from 6.76 to 6.78 and nothing else.

## Deployment

The 24 GB winner beats Qwen3.5-4B, so by the protocol it was put through the
measurements of part 2 of `docs/RESPONDER_SWITCH_PROTOCOL.md` (whole-stack
memory within 22,528 MiB) as the responder of a `serve-live` preset
(`--responder gemma4_12b_qat_elderly_v1`): the vLLM serving configuration of
the decision (context 2,048, two sequences, the profiled KV cache of
1,481,617,055 bytes), elderly prompt v1, at most 96 new tokens, otherwise the
v2 live configuration. Receipts:
`results/eval/responder_v3_24gb_qualification.json` (with its loopback and
coherence receipts beside it).

| Check | Result | Rule | |
|---|---|---|---|
| Q1 whole-stack memory | 12,466 MiB (recognizer 1,610, vLLM 10,856) | ≤ 22,528 MiB | pass |
| Q2 latency, plain panel (201 turns) | no missing reply; mean 372 ms, p90 468 ms | mean and p90 ≤ 500 ms | pass |
| Q2 latency, elderly panel (80 turns) | p90 373 ms | p90 ≤ 500 ms | pass |
| Q3 relevance after interruption, Gemini | 2.37 vs control 2.73 (−0.37, 95% CI [−0.90, 0.13]) | at most 0.25 below | fail |
| Q3 relevance after interruption, Qwen3.8-27B | 2.50 vs control 2.87 (−0.37, [−1.10, 0.37]) | at most 0.25 below | fail |
| Q4 script | 0 of 419 replies with a Latin letter | none | pass |

Part 2 is therefore **not passed**, and the preset is not qualified. It stays
in `serve-live` as an opt-in configuration, and the default is unchanged. All
30 interrupted replies were playing at the onset and all stopped. Both Q3
intervals include zero, but the rule, frozen before this run, compares the
point estimates.

Where the deficit comes from (exploratory, after the outcome was known): it
lies in the late interruptions (relevance 2.0 vs 2.9 under Gemini, 1.9 vs 3.2
under Qwen3.8; early and mid are within 0.2). Reading the runs showed that
Gemma 4 12B asks the user to repeat when the recognizer garbles the follow-up,
where Qwen3.5-4B guesses. `scripts/count_clarification_requests.py` counts such
replies with a fixed pattern (`results/eval/responder_v3_repeat_requests.json`):
Gemma 4 12B asks in 14 of its 60 follow-up replies (4 of the 10 late
interruptions), Qwen3.5-4B in 3 of 60 on the same scenarios. The follow-up
transcripts Gemma asked about have a median character error rate of 0.345,
against 0.161 for the others. The written-request comparison above had clean
text input, so it could not show this. Asking to repeat may suit older users,
but the judges, who see the intended follow-up and not the recognizer's
transcript, score it as irrelevant. Whether people prefer it is a question for
human evaluation.

### 12 GiB: two post-hoc checks of Gemma 4 12B (neither passed)

The 12 GiB profile has no winner. Two post-hoc addenda, written after that
result and before their measurements, checked whether Gemma 4 12B could still
serve the 12 GiB stack. Both ran the same four checks with the 12,288 MiB limit,
on the same harness, panels and host.

**vLLM QAT with one sequence** (post-hoc addendum of 2026-11-11; receipts
`results/eval/responder_v3_12gib_one_sequence_qualification*.json`). One
sequence with a KV cache of 749,197,135 bytes brings the vLLM build within the
limit.

| Check | Result | |
|---|---|---|
| Q1 memory | 11,768 MiB (recognizer 1,610, vLLM 10,158) | pass |
| Q2 latency | plain: 1 of 201 turns without a reply; mean 344 ms, p90 385 ms; elderly p90 387 ms | fail |
| Q3 relevance after interruption | Gemini −0.10 ([−0.63, 0.43]), Qwen3.8 −0.27 ([−1.07, 0.53]) | fail (Qwen3.8) |
| Q4 script | 0 of 416 replies with a Latin letter | pass |

The turn without a reply is a 1.2 s utterance for which the server recorded no
turn at all: the end-of-speech detector, which runs before the responder, never
closed it. The same utterance was answered in the 24 GB run. The rule counts it.

**llama.cpp diagnosis** (exploratory; `scripts/diagnose_llamacpp_first_token.py`,
`results/eval/responder_v3_llamacpp_first_token_diagnosis*.json`). The
llama.cpp builds fit 12 GiB but missed the latency gate. The diagnosis served
Gemma 4 12B Q5_K_M with the harness settings on the first 40 development rows,
changing one setting at a time. Median first token:

| Setting | First pass (host load ≈ 2) | Second pass (host load 2–35) | GPU memory |
|---|---|---|---|
| harness settings | 184 ms | 280 ms | 9,766 MiB |
| quieter log | 180 ms | 183 ms | 9,766 MiB |
| no Persian-only grammar | 179 ms | 163 ms | 9,766 MiB |
| micro-batch 2,048 | 178 ms | 176 ms | 10,460 MiB |
| one slot | 177 ms | 175 ms | 9,254 MiB |
| no prompt cache | 138 ms | 136 ms | 9,768 MiB |
| no context checkpoints | | 59.5 ms | 9,766 MiB |
| full window cache (`--swa-full`) | | 24.8 ms | 10,088 MiB |

Gemma 4 uses sliding-window attention. With its defaults, llama.cpp saves and
restores context checkpoints of the window cache (50–140 MiB each) around
every reused prompt. That, not the model, the grammar or the GPU, cost the
first token about 150 ms. A full-size window cache keeps prompt reuse without
checkpoints.

**llama.cpp Q5_K_M with `--swa-full`** (post-hoc addendum 3; receipts
`results/eval/responder_v3_12gib_llamacpp_qualification*.json`; live preset
`gemma4_12b_q5km_llamacpp_elderly_v1`, with the Persian-only constraint as the
equivalent llama.cpp grammar).

| Check | Result | |
|---|---|---|
| Q1 memory | 11,700 MiB (recognizer 1,610, llama.cpp 10,090) | pass |
| Q2 latency | plain: no missing reply; mean 480 ms, p90 824 ms; elderly p90 1,005 ms | fail |
| Q3 relevance after interruption | Gemini −0.33 ([−0.90, 0.20]), Qwen3.8 −0.33 ([−1.07, 0.40]) | fail |
| Q4 script | 0 of 409 replies with a Latin letter | pass |

The first 125 plain turns had a median of about 365 ms. From about turn 150
every stage slowed at once, the recognizer too (55 → 75 ms, a separate
process), the first token (105 → 180 ms) and the first clause (180 → 580 ms).
That points to load from other users of the shared host, which this run did not
record. As the rule is written, Q2 fails as measured. Q3 fails under both
judges either way, so it was not repeated.

**The common cause.** All three Gemma 4 12B runs lose relevance after late
interruptions (Gemini 2.0, 2.5 and 1.9 against controls of 2.9 to 3.1), and in
all three the model asks the user to repeat after a garbled follow-up
(`results/eval/responder_v3_repeat_requests.json`, exploratory):

| Run | Follow-ups asking to repeat (interrupt + control, of 60) | Follow-up CER when asking / otherwise |
|---|---|---|
| Qwen3.5-4B, vLLM (switch qualification) | 3 | 0.207 / 0.200 |
| Gemma 4 12B QAT, vLLM, 24 GB | 14 | 0.345 / 0.161 |
| Gemma 4 12B QAT, vLLM, 12 GiB one sequence | 13 | 0.351 / 0.146 |
| Gemma 4 12B Q5_K_M, llama.cpp, 12 GiB | 11 | 0.420 / 0.167 |

So no Gemma 4 12B configuration is qualified for either stack, and the 12 GiB
stack keeps the responders the listening study decides between. Changing how
the responder treats an uncertain transcript (the recognizer's confidence, or
an instruction to answer the most likely request and confirm briefly) is the
next step, and it needs its own pre-registered test.

## Claim boundary

Automatic judges only (two families with no candidate; agreement with people
is not measured here); written requests and caption turns, not spoken
interaction; latency on the shared H100 (other users' load slowed some
llama.cpp runs, hence Amendment 3's third measurement), not on a 12 or 24 GB
card; community builds as published on the inventory date.
