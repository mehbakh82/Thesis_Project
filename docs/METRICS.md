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

## Automatic synthesized-speech CER/WER proxy

For content-controlled cascade speech, the generated reply text is the
reference and the real NeMo transcription of its Piper waveform is the
hypothesis. Both are passed through the deterministic Persian verbatim
normalizer. Word tokens split on whitespace and treat ZWNJ as a boundary;
character tokens are Unicode code points excluding whitespace and ZWNJ.

Report per-row and macro mean/p50/p95/maximum error rates. Micro WER/CER are the
sum of Levenshtein edits divided by the summed reference word/character counts.
ASR failures stay in the denominator and invalidate the frozen measurement.

This is an `automatic_asr_roundtrip_intelligibility_proxy`, not
`official_e2e`, semantic, pronunciation, naturalness, human-intelligibility, or
population evidence. The same ASR family participates in the system and proxy,
and the nine single-voice outputs do not justify a confidence interval or an
after-the-fact quality threshold. See
`docs/CASCADE_INTELLIGIBILITY_PROTOCOL.md`.

## Automatic semantic responder final test

The prompt-v2 protocol fixes a two-stage design. Seven previously unused
validation rows form the eligibility stage; only a complete pass unlocks 40
metadata-preselected test rows across all five held-out source sessions.
Three blinded arms (Qwen2.5-0.5B baseline, Qwen3-4B prompt-v2 candidate, and
automatically aligned next-speaker reference) receive two deterministic judge
calls for relevance and coherence on a 0–4 scale.

The predeclared candidate gates are: relevance mean ≥2.0, coherence mean ≥2.5,
relevance gain over baseline ≥0.5, nonnegative coherence gain, paired relevance
win rate ≥0.60, relevance-at-least-two rate ≥0.70, and complete mechanical
validity. Failed calls stay in the denominator and invalidate the result.

This evidence class is
`automatic_same_family_llm_as_judge_qwen4b_prompt_v2_final_test`. It is not
human semantic evaluation, an independent dialogue benchmark, factuality or
safety evaluation, naturalness evidence, population evidence, physical-4090
evidence, or official browser latency. See
`docs/QWEN4B_CASCADE_V2_PROTOCOL.md`.

## Barge-in accuracy

Provisional primary metric: event-level interrupt vs other **accuracy ≥ 0.80** on speaker/session-held-out real interactions. Also report interrupt F1, FAR, FRR, denominators, and 95% confidence intervals. Energy-only VAD is the baseline. Confirm event-level versus frame-level interpretation with the supervisor.
