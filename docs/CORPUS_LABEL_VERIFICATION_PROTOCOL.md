# Corpus label verification — protocol

Status: written before any verification call. Purpose: measure how often the
automatic labels of the conversation corpus agree with an independent automatic
listener. The corpus turn boundaries come from diarization plus caption timing,
and its overlap labels from diarized overlap intervals; neither was checked by a
person (the QA waiver in `docs/QA_WAIVER.md`). Gemini 3.8 Flash gives
**independent automatic labels**, never human verification.

## Sample

- **Frame:** balanced-v2 pairs (`data/processed/manifests/conversations_balanced_v2.jsonl`,
  6,551 pairs) whose source window audio is still on local disk after the
  2026-10-06 audio offload: 1,353 pairs (Digiato 956, Zoomit 281, Mehran
  Rowshan 107, Tabaghe16 9). The frame is a convenience frame; its channel mix
  differs from the corpus and is reported.
- **Strata:** the pipeline's overlap label at the turn boundary. A pair is
  *boundary-overlap* when one of its overlap intervals intersects the 2 s
  around the boundary (the end of the user turn), otherwise
  *boundary-clean*. 150 pairs are drawn from each stratum (all of a stratum if
  it has fewer), seed 20261018, after sorting by `utt_id`.
- Corpus-level rates are weighted by the strata shares in the full 6,551-pair
  manifest (computed from the manifests, without audio).

## Labels per pair (three Gemini calls)

1. **Boundary, two passes.** An 8 s clip centred on the boundary, labeled with
   the two independently worded prompts of the detector v2 study
   (`scripts/label_interaction_events.py`, passes a and b): interrupt,
   backchannel, turn_change, noise or none, plus whether two voices overlap.
   A pass agrees with the pipeline's turn boundary when a new voice starts
   there (interrupt, backchannel or turn_change).
2. **Transcript.** User turns of at most 60 s are transcribed verbatim in full
   with the project's verbatim prompt, and the caption text of the turn is
   scored against it with the `fa-asr-score-v1` normalizer; longer turns are
   left out of this measure (their number is reported).

Only labels on which both boundary passes agree enter the rates; the
two-pass agreement is reported.

## Measures (Wilson 95% intervals)

- **Boundary speaker-change rate:** share of agreed boundaries where a new
  voice starts, per stratum and weighted.
- **Overlap agreement:** among agreed boundaries, the share where Gemini hears
  overlap in the boundary-overlap stratum (precision of the overlap label) and
  hears none in the boundary-clean stratum.
- **Caption accuracy:** corpus CER and median per-turn CER of the captions
  against the Gemini transcripts, and the share of turns with CER ≤ 20%.

No threshold is set in advance: this is a measurement of label quality, not a
pass/fail gate, and the corpus is not changed by it.

## Claim boundary

Automatic labels checked by another automatic system on a convenience frame of
the corpus; no human annotation.
