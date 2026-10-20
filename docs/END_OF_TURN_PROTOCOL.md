# Semantic end-of-turn gate — protocol

Status: frozen on 2026-10-20, before any candidate below was scored on any
audio. The only measurement made before freezing is the no-gate baseline on the
development clips (stated below). Code: `src/thesis_s2s/runtime/turn_end.py`
(gates), `Endpointer.hold` (`runtime/vad.py`), `scripts/end_of_turn_dev.py`
(development replay and selection).

## Problem

The deployed endpointer is acoustic: after 96 ms of silence it starts a
speculative reply (audio held), and after 256 ms of silence it confirms the end
of the turn and releases the reply. A speaker who pauses for longer than
256 ms in the middle of a sentence is cut off and answered on a fragment. In
live loopback v2 this happened to 15 of 201 plain and 7 of 80 simulated-elderly
planned utterances (`docs/LIVE_LOOPBACK_V2_RESULT.md`, amendment 2); slower,
more hesitant speech makes it more frequent.

Public end-of-turn models exist, but none covers Persian: Smart Turn v3
(pipecat, audio-only, 23 languages) and the LiveKit turn detector (text, 14
languages). This experiment tests whether one of them, or a Persian-specific
alternative, reduces premature endpoints without breaking the latency budget.

## Mechanism (identical for every candidate)

At the speculative start the gate scores the utterance so far and returns
P(complete). If P(complete) < θ, the current pause is **held**: the end of
the turn is confirmed only after H ms of total silence instead of 256 ms. If
the user resumes speaking within H, the speculative reply is cancelled and the
utterance continues, as already happens before 256 ms. If the gate's result is
not ready when the silence reaches 256 ms, the pause is not held (fail-open), so
the gate can add latency only through a hold. A hold that ends without
resumed speech delays that turn's first audio by H − 256 ms.

## Candidates

- **A — Smart Turn v3.2, zero-shot.** `smart-turn-v3.2-cpu.onnx` from
  `pipecat-ai/smart-turn-v3` (revision `f766f81d3cfdf7737ac64aad813d91bbfd56bf93`,
  BSD-2), on CPU with one thread, with the reference preprocessing (last 8 s,
  zero-padded at the start, Whisper features). Input: the audio handed to the
  speculative turn (from the utterance start with the deployed 200 ms pre-roll
  to the last voiced frame).
- **B — LLM completeness, zero-shot.** The deployed Qwen3-4B on the same vLLM
  server, given the speculative ASR transcript (no extra ASR call) and the
  fixed prompt in `src/thesis_s2s/runtime/turn_end.py`; P(complete) from the
  first-token probabilities of «1» (finished) and «0» (unfinished).
- **L — lexical rule.** Hold if the last word of the speculative transcript is
  one of a fixed list of Persian words that cannot end an utterance
  (conjunctions, prepositions, the object marker, determiners and discourse
  fillers; the list is in `turn_end.py`), or if the transcript is empty.
  P(complete) is 0 or 1.

Grid: θ ∈ {0.3, 0.5, 0.7} for A and B; H ∈ {640, 1024} ms for all.

## Development benchmark (offline, exact replay)

- **Clips:** the 425 single-speaker clips of 1–5 s in the barge-in speech pool
  whose sessions contain no clip of any loopback test panel (69 sessions), each
  in the plain condition and in the simulated-elderly condition of the test
  panels (ffmpeg `atempo=0.85,volume=-3dB`): 850 trials.
- **Replay:** each trial (0.3 s of silence before, 1.5 s after) is pushed
  through the deployed `Endpointer` (Silero, 256 ms hangover, 96 ms
  speculative start) with the gate in the loop. Gate compute time is measured
  on this host with the stack running (ASR service for B and L, vLLM for B)
  and applied: a result arriving later than 160 ms after the speculative start
  counts as not held.
- **Baseline (measured before freezing, with `scripts/end_of_turn_dev.py`):**
  without a gate, 78 of 425 plain and 90 of 425 simulated-elderly trials end
  prematurely (19.8% pooled).
- **Measures:** *premature-end rate* (trials with a confirmed end before the
  clip's last voiced frame), *false-hold rate* (trials whose final end was
  held), late-gate rate, and mean added delay at the final end.

## Selection rule (stage 1)

Among the configurations with a pooled false-hold rate of at most 5%, choose
the one with the lowest pooled premature-end rate. Ties within one percentage
point go to the lower false-hold rate, then the smaller H, then the cheaper
candidate (L, then A, then B). A gate is adopted only if it lowers the
premature-end rate by at least 25% relative to no gate; otherwise stage 2.

## Stage 2 (only if stage 1 adopts nothing)

Fine-tune Smart Turn v3.2 from its public checkpoint with its public training
code on Persian examples mined from **training-split sessions only** of the
thesis corpus: *incomplete* = audio of a diarized turn cut at an internal pause
of at least 100 ms after which the same speaker continues; *complete* = audio
ending at the end of a turn that is followed by a speaker change. Session-
held-out validation inside the training split; then the same development
benchmark and rule. These labels are automatic (diarization), not semantic
judgements.

## Test (once, with the selected gate)

Live loopback v3: the plain (201) and simulated-elderly (80) panels of v1/v2,
with the same harness, host, stack and scorer, the gate as the only change.

- **Primary:** planned utterances answered as more than one server turn,
  v3 against v2 per utterance (exact McNemar test, both panels pooled and per
  panel).
- **Latency:** as in v2 — mean, median, p90, p95, maximum, share ≤ 500 ms, and
  the audible p90.
- **Decision:** the gate stays in the deployed configuration if the pooled
  number of split utterances is lower than in v2 **and** the plain panel keeps
  mean and p90 ≤ 500 ms **and** the elderly p90 stays ≤ 500 ms. Otherwise it is
  removed and reported as a negative result.

The barge-in panels are not re-run: the gate acts only while the system is
listening and does not change barge-in decisions. This is the third
measurement of the same latency panels after a change chosen on separate
development audio and is reported as such, next to v1 and v2.

## Claim boundary

Automated loopback with recorded and tempo-slowed speech on the shared H100
host; simulated elderly speech, not real older speakers; development clips are
podcast segments whose ends are diarization boundaries, not always finished
sentences.

## Amendment 1 (2026-10-20, after the first stage-2 model, before the corrected one)

Stage 1 adopted nothing (`results/eval/end_of_turn_dev_stage1.json`): every
zero-shot candidate held more than 5% of the final ends (best false-hold rate
13.1%, lexical). Inspection of the held final ends showed that many development
clips end mid-sentence (most often in «که», «این», «رو», «یعنی»), so part of
every candidate's false-hold rate is correct holding of unfinished clip ends.

Stage 2 examples could not come from the diarized windows (after excluding the
development and test sessions only 6 windows remained); they were mined from
the user channel of the training-split conversation pairs (144 sessions,
9,308 training and 964 validation examples). The first fine-tuned model is
**invalid**: in the pairs the user channel is cut to digital silence where the
other speaker takes over, so its *complete* examples ended while the speaker
was still voicing (last 60 ms at a median −0.8 dB relative to the clip, against
−13.0 dB for *incomplete* examples cut at a pause). The model learned the cut,
not the turn end, and on the development trials held 92–97% of final ends. It
is reported only for that observation.

Correction (the label definition is unchanged; the cut is made the way the
live gate sees audio): both classes are cut at the last voiced frame before a
silence of at least 100 ms (Silero offset threshold 0.35, as the endpointer);
an example is *complete* only if the speaker stopped naturally, that is, at
least 100 ms of non-silent channel audio follows the last voiced frame of the
turn. The corrected model is trained with the same settings and scored once on
the same development trials with the same rule; the loopback test is opened
only if it is adopted.
