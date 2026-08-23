# Latency and interrupt metrics

## T_first_audio

Wall-clock milliseconds from **user end-of-speech** (or the moment a barge-in is accepted as a new turn) to the **first audible PCM sample** of the assistant reply.

The historical engineering contract used **p50 ≤ 500 ms**. The Persian project definition can be read literally as **maximum ≤500 ms**. These are not equivalent. Until the supervisor resolves this, report p50, p95, and maximum, label the p50 gate provisional, and separately report whether every measured turn is ≤500 ms.

Do not discard timeouts or failed turns to make the maximum pass. State the timeout policy and denominator.

This does **not** include the duration of the user utterance.

The official value is client-observed: the browser sends `playback_started` only when the first source is actually scheduled. Server generation time is useful diagnostics, but it is not this metric.

## T_barge_in

Wall-clock milliseconds from the **acoustic onset** of a user interrupt to the moment assistant **playback actually stops**.

Gate: **p95 ≤ 300 ms**.

The classical detector in `src/thesis_s2s/bargein` **owns** this control path (`PlaybackController.stop_playback`).

The official stop time comes from the browser's `playback_stopped_ack` after every active audio source has been stopped. A server-side detector timestamp alone is not sufficient.

## Evidence classes

- `official_e2e`: consented live browser session, client playback acknowledgements, real microphone audio, and eligible physical GPU.
- `live_unofficial`: same protocol, but hardware or study conditions do not meet the definition.
- `server_component`: server generation/detector timing without proof of audible client behavior.
- `synthetic_proxy`: generated audio or simulated events; valid for regression tests only.

Only `official_e2e` rows may be used to claim that a thesis latency gate was met.

## Hardware

Official tables require a physical GPU with **12–24 GB** VRAM. Training may use the H100 NVL, but a software memory cap does not make a 93 GB H100 an eligible card. Such runs must be labelled `memory_capped: true` and unofficial.

## Barge-in accuracy

Provisional primary metric: event-level interrupt vs other **accuracy ≥ 0.80** on speaker/session-held-out real interactions. Also report interrupt F1, FAR, FRR, denominators, and 95% confidence intervals. Energy-only VAD is the baseline. Confirm event-level versus frame-level interpretation with the supervisor.

