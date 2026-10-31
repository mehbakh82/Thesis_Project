# Semantic end-of-turn gate v2 — protocol

Status: frozen on 2026-10-28, before any clip-end label below was produced and
before the new stage-2 data were mined. Mechanism, candidates, development
trials, replay, grid and the loopback test are those of
`docs/END_OF_TURN_PROTOCOL.md`; this document changes only what is listed
here.

## Why a v2

The first run (`docs/END_OF_TURN_RESULT.md`) found two flaws that are about
data, not about the gates:

1. **The false-hold measure counted correct holds as errors.** The development
   clips end at diarization boundaries, and many end mid-sentence (most often in
   «که», «این», «رو», «یعنی»). Holding such an end is the right behaviour, but
   the rule counted it as a false hold. A clip set in which roughly 13% of ends
   are unfinished cannot pass a 5% bar, whatever the gate.
2. **Stage 2 had 738 training examples.** The 201 h of diarized windows with
   natural turn ends had been moved to S3 to save disk, so examples came from
   the conversation pairs, where 90% of turns are cut while the speaker is still
   voicing. The windows are restored for v2.

What is already known: the per-trial replays of every v1 configuration exist,
and the v1 inspection suggested that most of the lexical rule's holds were on
unfinished ends, so the lexical rule may pass once correct holds stop counting.
The development decision below therefore carries little independent weight.
**The decision that counts is the loopback test, whose panels are still
unopened.**

## Clip-end labels (development trials)

Each of the 425 development clips is labelled from its audio by Gemini 3.8
Flash (temperature 0) in two independent passes with different wordings; the
simulated-elderly trial of a clip takes the clip's label.

- **Pass A:** "You will hear a short Persian speech clip. Decide whether, at
  the very end of the clip, the speaker has finished what they were saying (a
  complete sentence or turn that a listener could answer now) or stopped in
  the middle (the sentence is unfinished and the speaker would naturally go
  on). Return only JSON: {"label": "complete"} or {"label": "incomplete"}."
- **Pass B:** "Listen to this Persian audio. Ignore the recording quality. Is
  the last utterance finished in grammar and meaning, or is it cut off or
  trailing (for example ending in a conjunction, a preposition, or an
  unfinished clause)? Return only JSON: {"finished": true} or {"finished":
  false}."

A clip end is *complete* or *incomplete* only if both passes agree; otherwise
it is *uncertain*. Agreement is reported. These are independent automatic
labels, not human labels. The labeller never sees a gate's output.

## Measures

- **Premature-end rate:** unchanged (all 850 trials).
- **False-hold rate (changed):** among trials whose clip end is labelled
  *complete*, the share whose final end was held.
- **Correct-hold rate (reported, not in the rule):** among trials whose clip
  end is labelled *incomplete*, the share held.
- The v1 false-hold rate over all trials is still reported next to it.

## Candidates

1. Every v1 configuration (L, A, B; θ ∈ {0.3, 0.5, 0.7}, H ∈ {640, 1024}),
   re-scored from its stored per-trial replay without re-running.
2. **Stage 2b — Smart Turn v3.2 fine-tuned on the restored windows.**
   - *Sessions:* training-split windows of the episode set and the Tabaghe16
     reserve, minus every development and loopback-test session (the v1
     exclusion list).
   - *Complete:* the end of a diarized single-speaker turn followed by a
     different speaker. The cut is the turn's last voiced frame (Silero,
     threshold 0.35). The next speaker must start at least 200 ms after the
     cut, and no overlap may occur in the turn's last second.
   - *Incomplete:* a pause of at least 100 ms inside a turn, after which the
     same speaker continues and no other speaker talks, at least 1 s after the
     turn start and 0.3 s before its end. The cut is at the pause onset. One
     pause per turn is chosen at random.
   - *Input:* the audio from 200 ms before the turn start to the cut, at most
     the last 8 s, as the live gate sees it.
   - *Label filter:* each example is transcribed by the deployed ASR, and
     Gemini judges each transcript, 40 per request, with this prompt: "Each
     item is an automatic transcript of a Persian speaker's speech up to a
     pause. For each item, decide whether the speaker had finished the sentence
     at that point (complete) or would clearly go on (incomplete)." An example
     is kept only if this label matches its diarization label.
   - *Training set:* classes balanced by down-sampling the larger one, at most
     24,000 examples, and 10% of sessions held out for validation.
   - *Training:* the v1 stage-2 settings (public training code, learning rate
     5e-5, 4 epochs, warm-up 0.2, weight decay 0.01, batch 64), exported to
     ONNX and checked against PyTorch.
   - *Scoring:* the v1 grid on the same development trials.

## Selection rule (unchanged except for the false-hold denominator)

Among configurations with a pooled false-hold rate of at most 5%, choose the
lowest pooled premature-end rate. Break ties within one percentage point by
the lower false-hold rate, then the smaller H, then the cheaper candidate. A
gate is adopted for the test only if it lowers the premature-end rate by at
least 25% relative to no gate.

## Test (once, with the selected gate)

Live loopback v3 exactly as in `docs/END_OF_TURN_PROTOCOL.md`:

- *Panels:* plain (201) and simulated-elderly (80), with the same harness,
  host, stack and scorer, and the gate as the only change.
- *Primary measure:* split utterances, v3 against v2 by exact McNemar test.
- *Decision:* the gate stays only if the pooled number of split utterances is
  lower than in v2, the plain panel keeps mean and p90 at or below 500 ms, and
  the elderly p90 stays at or below 500 ms.

If nothing is adopted, the panels stay closed.

## Claim boundary

Automatic clip-end labels from one labeller family, offline development
replay, simulated elderly speech, and loopback on the shared H100.
