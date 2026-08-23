# Evaluation evidence status

**Verdict: no thesis-level end-to-end gate is currently satisfied by collected evidence.** Historical JSON files are retained for provenance but are component/synthetic proxies.

## Barge-in detector

- Proposed GBDT: accuracy **1.0**, F1 1.0, FAR 0.0, FRR 0.0 (n=32)
- Energy-VAD baseline: accuracy 0.375, FAR 1.0
- Evidence class: **synthetic_proxy**. Recorded speaker/session-held-out evaluation is not collected, so the >80% thesis gate is pending.
- CPU hop: ~8 ms (budget ≪ 20 ms)

## Historical latency proxy

The old files timed server-side generation and simulated detector handling, not first audible browser playback and acknowledged browser stop. They also exercised a legacy `OmniTalker` that Piper bypassed; that checkpoint is now rejected by the runtime.

| Historical file | Hardware | Evidence class | Reported proxy |
|---|---|---|---|
| `latency_bench.json` | H100 NVL with 24 GB software cap | server/synthetic component, unofficial | 60 ms generation p50 / 18 ms detector p95 |
| `latency_bench_path_b.json` | 93 GB H100 NVL uncapped | server/synthetic component, unofficial | 45 ms generation p50 / 16 ms detector p95 |

Neither row may be used to claim the ≤500 ms or ≤300 ms end-to-end gate. A memory cap does not turn an H100 into an eligible physical 12–24 GB GPU.

Historical cascade server observations were ~1.1–2.4 s and already exceeded the target. Re-measure the corrected complete-reply cascade through the live browser protocol.

## Speech quality

`reply_wer.json` has a mean WER of 104 over only three prompts. It is a failed/diagnostic ASR-on-TTS proxy, not evidence of acceptable naturalness or intelligibility. No valid MOS table has been collected.

## Human study

`human_study.json` is **incomplete**: one participant directory, one partial rating, zero turns, zero complete ratings, and no client timing. Required: 5–10 participants, at least two aged 60+, complete ratings, real detector evidence, and eligible hardware.

## Current use

Run `serve --record --study` for collection, then `export-recordings` and `study-summary`. The authoritative artifact classification is `results/eval/EVIDENCE_STATUS.json`; metric semantics are in `docs/METRICS.md`.
