# Live latency and barge-in loopback — formal result

Protocol: `docs/LIVE_LATENCY_LOOPBACK_PROTOCOL.md` (frozen before the formal
runs; amended before them for the independent backchannel panel; scoring
correction documented there). Receipt: `results/eval/live_loopback_formal.json`
(per-condition receipts and server telemetry kept in the work directory; their
hashes are in the receipt).

## Setup

Headless Chromium drives the real client page with a fake microphone file;
audio takes the browser path (Web Audio capture → 20 ms frames → WebSocket →
server → streamed clauses → Web Audio playback). Stack (from `/health`): NeMo v4
CTC ASR, Qwen3-4B-Instruct-2507 prompt-v2 on vLLM with a Persian-only
constraint, Mana Piper on CPU, Silero endpointing (speculative 160 ms, confirm
256 ms), barge-in controller as selected by the simulation (one-frame duck,
double-talk margin 14 dB, coupling prior 0 dB, lexical confirm v2).

**Memory budget:** the LLM server was capped at 0.105 of the H100 (2048-token
context, two sequences) so that the GPU side of the stack fits a 12 GB card.
Measured peak during all runs: **10,994 MiB = 10.7 GiB** (LLM 9,388 MiB reserved + ASR 1,606 MiB), within a 12 GiB (12,288 MiB) card;
TTS and VAD run on CPU. Compute speed is that of the H100, not of a 12–24 GB
GPU (disclosed deviation).

## End of speech → first assistant audio (client clock)

| Panel | Planned turns | Missing | Mean | Median | p90 | p95 | Max | ≤ 500 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Plain (61 PSRB + 140 corpus utterances, 1–5 s) | 201 | 0 | **418** | 415 | **487** | 506 | 537 | 93.5% |
| Simulated elderly (0.85× tempo, −3 dB) | 80 | 0 | 435 | 433 | 508 | 525 | 575 | 86.3% |
| Barge-in panels (base turns) | 120 | 0 | 399 | 388 | 464 | 472 | 509 | 99.2% |

T_first_audible (adding the browser's reported output latency): plain median
457 ms, p90 529 ms.

Server-side stages from the speculative start (plain, medians): ASR 62 ms,
first LLM token 102 ms, first clause 177 ms, first clause synthesized 231 ms;
plus the 160 ms of silence before speculation starts. 216 of 226 speculative
starts were confirmed.

## Barge-in (60 turns with a real interruption, 60 with a real backchannel)

| Measure | Value |
|---|---|
| Interruptions stopped | 50/60 (83.3%) |
| Backchannels wrongly stopped | 8/60 (13.3%) |
| Pooled accuracy / F1 (interrupt = positive) | **0.850 / 0.847** |
| Detected onset → client duck: median / p95 / max | **46** / 61 / 62 ms |
| Detected onset → client stop: median / p90 | 493 / 851 ms |

T_duck here starts at the frame where the server detected the onset; the
latency from the true acoustic onset is measured in the simulation (median
120 ms, `docs/BARGEIN_SIMULATION_RESULT.md`).

## Acceptance (proposal criteria, as frozen)

| Criterion | Result |
|---|---|
| Mean end-to-end latency ≤ 500 ms (plain) | **Met** — 418 ms |
| 90th percentile ≤ 500 ms (plain, missing turns counted) | **Met** — 487 ms |
| Interruption detection ≥ 80% (accuracy and F1) | **Met** — 0.850 / 0.847 |
| Detect and adapt within 150 ms (median duck) | **Met** — 46 ms from detected onset (120 ms from true onset in simulation) |

Reported but not pass criteria: the maximum (537 ms) and p95 (506 ms) exceed
500 ms; the simulated-elderly p90 is 508 ms; the audible p90 is 529 ms.

## Claim boundary

Automated browser loopback with pre-recorded utterances on the shared H100
host, with the GPU memory of a 12 GB card but not its compute; no human user and
no physical microphone/speaker path (no acoustic echo). Not the proposal's
physical target-hardware condition.
