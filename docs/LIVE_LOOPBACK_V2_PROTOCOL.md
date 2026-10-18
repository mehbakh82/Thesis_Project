# Live loopback v2 — protocol

Status: written before any v2 run. Two changes to the deployed live system and
one new measurement, on the same stack, host and harness as the formal loopback
(`docs/LIVE_LATENCY_LOOPBACK_PROTOCOL.md`, receipt
`results/eval/live_loopback_formal.json`, called v1 below).

## Change 1 — earlier speculative start (latency)

In v1 the endpointer starts a *speculative* turn after 160 ms of silence and
confirms the end of the turn after 256 ms; audio is held until confirmation.
The first clause is ready about 390 ms after the end of speech, well after the
confirmation, so starting the speculative turn earlier shortens the time to
first audio without changing when a turn is confirmed or cut (confirmation
stays at 256 ms; a speculative turn whose user keeps talking is cancelled).

- **Candidates:** speculative start at 64, 96, 128 or 160 ms.
- **Development panel:** 60 utterances of 1–5 s from the barge-in speech pool
  that are in none of the v1 panels (seed 20261018), plain condition, one run
  per candidate.
- **Rule:** the candidate with the lowest development p90 of T_first_audible
  (first audio plus the browser's reported output latency); if a larger
  candidate is within 10 ms of it, the larger one (fewer speculative starts).
  Missing turns count as failures; a candidate with a missing turn is
  excluded. Speculative starts and cancellations per turn are reported.
- **Evaluation:** the v1 panels (plain 201, simulated elderly 80, barge-in 60 +
  60) re-run once with the chosen value and scored with the v1 scorer. This is
  a second measurement of the same test panels after a change chosen on
  separate development audio, and is reported as such, next to v1.
- **Acceptance (as in v1, plus the audible reading):** mean and p90 of
  T_first_audio ≤ 500 ms (plain); additionally reported against 500 ms:
  p90 of T_first_audible (plain and elderly), the maximum, and the share of
  turns ≤ 500 ms; barge-in accuracy and F1 > 0.80; median duck ≤ 150 ms.

## Change 2 — heard-only history after an interruption

In v1 an interrupted reply entered the conversation history in full, although
the user heard only its beginning. Now the client reports in `stop_ack` how
much of the reply reached the speaker before the user started talking, and the
history keeps only those words, ending in «…» (`heard_prefix` in
`src/thesis_s2s/runtime/live.py`, with tests). Change 2 is in place for every v2
run.

## Measurement — response coherence after an interruption

The proposal's section 6.4 asks for interruptions at early, mid and late
timings and for the coherence of the response.

- **Scenarios:** 30 elderly requests from the responder-v2 elderly panel, each
  with a follow-up that only makes sense in its context (a correction or added
  detail), written before any run (`configs/interruption_coherence_scenarios.jsonl`).
  The interruption starts 0.8 s (early), 1.8 s (mid) or 3.0 s (late) after the
  request ends, rotating over the scenarios (10 each).
- **User voice:** the male Piper voice `fa_IR-amir-medium` at 0.85× tempo (as in
  the listening-study demos); synthesized speech, not a human talker.
- **Conditions, one fresh browser session per scenario:**
  *interrupt* — the follow-up is spoken over the reply to the request;
  *control* — the same follow-up is asked as a normal second turn after the
  reply has finished.
- **Measures:** stop rate (interruptions that stop the reply, among scenarios
  where the reply was still playing at the interruption onset; the others are
  reported separately); the transcript CER of the follow-up; and two judges
  (Gemini 3.8 Flash and the local Qwen3.8-27B) scoring the reply to the
  follow-up 0–4 for relevance to the follow-up given the first request, use of
  the first request's context, and coherence. The judges see the two user
  texts as written and the reply; they do not see the condition.
- **Criterion:** post-interruption replies are coherent if their mean
  relevance is at most 0.25 below the control under both judges (point
  estimate; paired bootstrap 95% intervals reported). Results are reported per
  timing.

## Claim boundary

Automated browser loopback with recorded or synthesized speech on the shared
H100 host, with the GPU memory of a 12 GB card but not its compute; automatic
judges; no human user.

## Amendment 1 (2026-10-18, while the v2 formal runs were in progress)

The thesis lists one more barge-in improvement as future work: the lexical
confirm transcribes the microphone from the detected onset, and because the duck
follows the first voiced frame, the first phoneme is often cut («آره» heard as
«کار»). Change 3 gives the confirm ASR 200 ms of audio before the onset
(`lexical_pre_roll_ms`, default off until decided).

- **Development:** the barge-in simulation's development stage on training
  audio, exactly as development run 6, with the families `lexical2f1` (the
  deployed controller) and `lexical2f1pre` (plus the pre-roll), double-talk
  margin 14 dB; the simulation's selection rule (highest accuracy, F1 next,
  lowest median duck within 0.01) decides.
- **If the pre-roll is chosen:** the simulation test split is re-run once with
  it (`results/eval/bargein_simulation_v2.json`; a second look at the test
  split after a change chosen on training audio, reported next to the first),
  and the two loopback barge-in panels and the coherence scenarios are run
  again with it (latency panels are unaffected: the pre-roll changes only the
  confirm decision). Otherwise nothing changes.
- These runs start only after the v2 runs end, so that they do not share the
  ASR service with a latency measurement.
