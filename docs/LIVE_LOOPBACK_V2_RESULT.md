# Live loopback v2 — result

Protocol: `docs/LIVE_LOOPBACK_V2_PROTOCOL.md` (with amendment 1). Receipts:
`results/eval/live_loopback_v2.json` (formal re-run), `live_loopback_v2_failures.json`
(failure modes); development choice under `results/eval/live_loopback_v2_dev_choice.json`.
v1 is `results/eval/live_loopback_formal.json` (`docs/LIVE_LATENCY_LOOPBACK_RESULT.md`).
Same host, stack, harness and scorer as v1.

## Change 1 — speculative start (development, 60 unused utterances)

| Speculative start | Missing | Mean | p90 | Max | Audible p90 | Speculative starts / turn |
|---|---:|---:|---:|---:|---:|---:|
| 64 ms | 0 | 360 | 442 | 501 | **484** | 1.27 |
| **96 ms (chosen)** | 0 | 383 | 446 | 531 | **488** | 1.23 |
| 128 ms | 0 | 413 | 471 | 560 | 513 | 1.20 |
| 160 ms (v1) | 0 | 424 | 499 | 556 | 541 | 1.20 |

Rule: lowest audible p90; a larger candidate within 10 ms is preferred, so
96 ms. Confirmation of the end of a turn stays at 256 ms, so endpointing
decisions are unchanged; only the work starts earlier.

## Evaluation — the v1 panels measured again with 96 ms

| Panel | Run | Turns | Missing | Mean | Median | p90 | p95 | Max | ≤ 500 ms | Audible p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain (61 PSRB + 140 corpus) | v1 | 201 | 0 | 418 | 415 | 487 | 506 | 537 | 93.5% | 529 |
| | **v2** | 201 | 0 | **372** | 375 | **431** | 447 | **491** | **100%** | **473** |
| Simulated elderly | v1 | 80 | 0 | 435 | 433 | 508 | 525 | 575 | 86.3% | 550 |
| | **v2** | 80 | 0 | **382** | 371 | **446** | 466 | 546 | 98.8% | 488 |
| Barge-in base turns | v1 | 120 | 0 | 399 | 388 | 464 | — | 509 | 99.2% | 506 |
| | **v2** | 120 | 0 | **366** | 363 | **415** | — | 497 | 100% | 457 |

The plain panel now meets even a strict every-turn reading (maximum 491 ms);
with the browser's reported output latency (about 42 ms) the p90 is 473 ms and
98% of turns are at or below 500 ms. Server stages from the speculative start
(plain, medians): ASR 65 ms, first token 42 ms later, first clause complete
85 ms later, its audio 56 ms later. Peak GPU memory of the stack: 11,336 MiB
(11.1 GiB; LLM reservation 9,730 MiB + ASR 1,606 MiB), within a 12 GiB card.

This is a second measurement of the same test panels after a change chosen on
separate development audio; v1 stays reported.

## Barge-in in the browser (v2, no confirm pre-roll)

| Measure | v1 | v2 |
|---|---:|---:|
| Interruptions stopped | 50/60 | 47/60 |
| Backchannels wrongly stopped | 8/60 | 8/60 |
| Pooled accuracy / F1 | 0.850 / 0.847 | 0.825 / 0.817 |
| Detected onset → duck, median / p90 | 46 / 60 ms | 44 / 60 ms |

The difference (three more missed interruptions of 60) is within sampling
noise; both runs meet accuracy and F1 > 0.80.

## Failure modes (v2)

- **Slow turns:** 0 of 201 plain turns and 1 of 80 elderly turns over 500 ms
  (v1: 13 and 11). In every slow turn, of v1 and v2, the stage furthest above
  its median was the completion of the first clause — token generation stalls
  on the shared GPU — not ASR or synthesis.
- **Missed interruptions (13):** 4 never ducked (no frame louder than the
  expected echo), 9 ducked and then resumed because the confirm transcript held
  no content word.
- **False stops on backchannels (8):** the confirm ASR heard a listener token
  as a content word (e.g. «آره آره» as «کار», its first phoneme cut); the
  remedy tested in amendment 1 did not help.

## Amendment 1 — confirm pre-roll (not adopted)

Barge-in simulation, development stage on training audio, 300 scenarios
(`results/eval/bargein_simulation_dev7_preroll.json`):

| Controller | Accuracy | F1 | Stop rate: interrupt / backchannel / noise / none |
|---|---:|---:|---|
| **Deployed (`lexical2f1`)** | **0.913** | **0.856** | 0.95 / 0.12 / 0.09 / 0.08 |
| + 200 ms confirm pre-roll | 0.900 | 0.840 | 0.98 / 0.14 / 0.15 / 0.10 |

The deployed controller reproduces its earlier development result exactly.
The pre-roll stops more interruptions but also more backchannels and noise
bursts (the audio before the onset carries echo and noise), so the selection
rule keeps the deployed controller; the test split was not re-run and the
browser barge-in numbers above are final.

## Amendment 2 — the first coherence run

The first coherence run used synthesized user lines with Piper pauses at «،»
and mid-line «؟» (median longest internal pause 380 ms); the endpointer closed
many requests mid-sentence (20 of 30 interrupt and 19 of 30 control sessions
had more turns than scripted), so that run is not scored. The same observation
on natural speech is a real failure mode: 15 of 201 plain and 7 of 80 elderly
planned utterances produced more than one server turn in v2. The stimuli were
fixed (no internal pause marks, internal pauses longer than 160 ms shortened to
120 ms) and the scenarios were run once more with the final configuration.

## Coherence after an interruption (final configuration)

Receipt `results/eval/interruption_coherence_v2.json`; 30 scenarios, one
browser session per scenario and condition; every session produced exactly the
two scripted turns.

| Measure | Result |
|---|---|
| Interruption landed while the reply was playing | 27 of 30 (3 replies had already ended) |
| Reply stopped | **24 / 27 (89%)**: early 10/10, mid 10/10, late 4/7 |
| Words of the interrupted reply kept in the history (median) | 3 |
| Follow-up transcript CER (median, synthesized voice) | 19% |
| Missing follow-up replies | 0 (interrupt), 0 (control) |

Judged reply to the follow-up (0–4; interrupt vs control, paired difference with
95% bootstrap interval):

| Judge | Relevance | Context use | Coherence |
|---|---|---|---|
| Gemini 3.8 Flash | 1.67 vs 1.67 (0.00, −0.43…+0.43) | 1.50 vs 1.50 (0.00) | 2.23 vs 2.30 (−0.07) |
| Qwen3.8-27B | 2.90 vs 2.80 (+0.10, −0.43…+0.67) | 3.00 vs 2.83 (+0.17) | 3.03 vs 3.03 (0.00) |

**Criterion met** under both judges: a reply after an interruption is as
relevant as the reply to the same follow-up asked as a normal turn. The
absolute level is the responder's (low under Gemini in both conditions, as in
responder v2), and the late scenarios score lower in both conditions alike.

## Claim boundary

Automated browser loopback with recorded or synthesized speech on the shared
H100 host, with the GPU memory of a 12 GiB card but not its compute; automatic
labels and judges; no human user.
