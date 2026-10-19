# Barge-in deployment simulation — result

Protocol: `docs/BARGEIN_SIMULATION_PROTOCOL.md`. Receipt (write-once):
`results/eval/bargein_simulation_final.json`. Development receipts dev1–dev6
used training-split audio only; the test split was run once.

## Deployed controller

`lexical2f1_margin_12.0_dtd_14.0`: Silero VAD; one voiced frame that exceeds
the expected residual echo by 14 dB (playback-level reference, coupling prior
0 dB adapting while the user is quiet) ducks the assistant; the deployed NeMo
v4 CTC ASR then transcribes the captured speech (at 400 ms of speech, every
200 ms after, and at 320 ms of silence) and the turn is stopped only if a word
remains after removing listener tokens, closed-class function words, and words
of the reply being played (exact, one edit, or a 3+-letter prefix/suffix).
Backchannel clips were selected by a different ASR (SeamlessM4T v2), so the
confirm is not circular.

## Test result (600 scenarios, 150 per event kind, early/mid/late onsets)

| Measure | Value (95% Wilson interval) |
|---|---|
| Accuracy (interrupt = positive) | **0.933** (0.910–0.951) |
| F1 | **0.880** |
| Recall: interruptions stopped | 0.980 (0.943–0.993) |
| Precision | 0.799 (0.735–0.850) |
| Stop rate on backchannels | 0.087 (0.051–0.143) |
| Stop rate on noise bursts | 0.093 (0.056–0.151) |
| Stop rate with no event (echo + background only) | 0.067 (0.037–0.118) |
| Interruptions stopped by onset: early / mid / late | 0.94 / 1.00 / 1.00 |
| Scenarios with a false duck before the event | 0.09 |
| Onset → duck: median / p90 / p95 / max | **120** / 247 / 327 / 985 ms (66% ≤ 150 ms) |
| Onset → stop: median / p90 / max | 786 / 1344 / 2023 ms |

Confusion: TP 147, TN 413, FP 37, FN 3.

## Against the proposal

- **Interruption detection ≥ 80%:** met on this benchmark — accuracy and F1
  both exceed 0.80 (the project requires both).
- **Detect and adapt within 150 ms:** the median onset-to-duck time is 120 ms;
  one third of interruptions take longer, so the criterion holds for the
  median, not for every event. Stopping the reply takes longer (median
  786 ms) because the controller waits until words can be recognized; during
  that time the assistant is already ducked.

## How the design got here (development split only)

| Development run | Change | Dev accuracy / F1 |
|---|---|---|
| dev3 | duration/MLP confirm + double-talk detector | 0.687 / 0.624 |
| dev4 | lexical ASR confirm with echo rejection | 0.840 / 0.765 |
| dev5 | + coupling prior, wider echo matching, function words | 0.920 / 0.862 |
| dev6 | + one-frame duck (latency tie-break) | 0.913 / 0.856, duck 122 ms |

## Claim boundary

Simulated mixtures of real recordings (real Persian speech, real
backchannels, real noise, real room responses; the residual echo is
attenuated, delayed assistant speech), not live users. Labels come from the
mining rules and an ASR listener plus the lexicon, not from humans. The browser
loopback (`docs/LIVE_LATENCY_LOOPBACK_PROTOCOL.md`) measures the same
controller end to end in a real browser.
