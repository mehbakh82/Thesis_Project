# Live latency and barge-in loopback — protocol

Status: frozen before the formal runs. Development runs (smoke and 40-turn
tuning runs) are reported separately and are not the formal result.

## What is measured

The hands-free streaming server (`thesis-s2s serve-live`) is driven by headless
Chromium through its real client page, with a fake microphone that plays a WAV
of pre-recorded Persian utterances (`scripts/loopback_latency.py`), as in the
proposal's section 6.4 ("loopback measurement using pre-recorded Persian
utterances of 1–5 s"). The client measures, on its own clock:

- **T_first_audio:** from the captured 20 ms frame holding the acoustic end of
  speech (server endpointer's last voiced frame) to the scheduled start of the
  first synthesized clause; **T_first_audible** adds the browser's reported
  output latency.
- **T_duck / T_stop:** from the captured frame holding the detected onset of a
  user interruption to the client's duck and stop acknowledgements.

## Stack (identity recorded in `/health` and in each receipt)

NeMo v4 CTC ASR (selected by ASR v2) → Qwen3-4B-Instruct-2507 served by vLLM
with a Persian-only regex constraint → clause-level Mana-Piper (bounded ONNX
threads) → two-stage barge-in controller with echo-floor gating. Endpointing:
Silero VAD, 256 ms hangover, speculative start at 160 ms, first fragment of 2–4
words.

## Panels

1. **Plain:** all 61 PSRB clips of 1–5 s (13 senior speakers) plus 140
   single-speaker corpus utterances (1.2–4 s) mined from held-out sessions —
   201 turns, 10 s between turns.
2. **Simulated elderly:** 80 of those utterances slowed to 0.85× tempo (pitch
   preserved) with a 3 dB lower level, modelling slower, quieter speech.
3. **Barge-in:** 60 turns with a second utterance (interruption, 1.2–3 s)
   starting 0.9 s after the turn ends, and 60 turns with an isolated real
   backchannel token at the same offset.

### Amendment before the formal runs (2026-10-17)

The server now confirms barge-ins lexically with the deployed NeMo ASR
(`docs/BARGEIN_SIMULATION_PROTOCOL.md`). The backchannel panel originally used
the 49 clips that NeMo itself had selected as listener tokens, which would make
the confirm circular; it is replaced by 60 clips drawn at random (seed
20261017) from the pool selected by an independent listener (SeamlessM4T v2).
For the barge-in panels the decision outcome is scored as well: a turn with an
interruption counts as correct when playback was stopped, a turn with a
backchannel when the reply completed.

## Acceptance (proposal criteria)

The proposal states **average ≤ 500 ms** (objective table) and **90th
percentile ≤ 500 ms** (problem statement) for end-to-end latency, and
interruption detection **within 150 ms**. Both latency criteria are evaluated
on T_first_audio of the plain panel; the stricter maximum is reported but not
used as the pass criterion (student decision recorded in
`docs/SUPERVISOR_DECISIONS.md`). Interruption detection latency is T_duck (the
"detect and adapt" stage); T_stop is reported separately. Every turn with a
missing acknowledgement or a server error counts as a failure in the
denominator.

## Claim boundary

This is automated loopback on the shared H100 host (CPU-side Piper and VAD),
not a human session and not a physical 12–24 GB GPU; the whole-stack GPU memory
is reported so the fit to a 12–24 GB device can be assessed, but compute speed
on such a device is not measured (disclosed deviation).
