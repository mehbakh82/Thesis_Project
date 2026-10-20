# Semantic end-of-turn gate — result

Protocol: `docs/END_OF_TURN_PROTOCOL.md` (with amendment 1). Receipts:
`results/eval/end_of_turn_dev_stage1.json` (zero-shot candidates) and
`results/eval/end_of_turn_dev.json` (all candidates, stage 2 included, final
decision). **Outcome: no gate met the pre-registered bar; the deployed
endpointer is unchanged and the loopback test panels were not opened.**

## Development benchmark

850 trials (425 single-speaker podcast clips of 1–5 s from 69 sessions that
appear in no test panel, each plain and simulated-elderly), replayed through
the deployed endpointer with the gate in the loop and measured gate timing.
Without a gate, 19.8% of trials are cut before their last voiced frame
(78/425 plain, 90/425 simulated elderly). The rule required a false-hold rate
(final ends held longer) of at most 5% and a reduction of at least 25%.

Best configuration of each candidate (all six or two grid points are in the
receipts):

| Candidate | θ | Hold | Premature ends | False holds | Mean added delay |
|---|---:|---:|---:|---:|---:|
| none (deployed) | — | 256 ms | 19.8% | 0% | 0 ms |
| L — continuation-word rule | — | 640 ms | 14.7% | 13.1% | 50 ms |
| A — Smart Turn v3.2, zero-shot | 0.3 | 640 ms | 15.2% | 25.8% | 99 ms |
| B — Qwen3-4B completeness, zero-shot | 0.3 | 640 ms | 13.3% | 15.4% | 59 ms |
| Stage 2 — Smart Turn v3.2 fine-tuned on Persian | 0.3 | 640 ms | 2.4% | 65.3% | 251 ms |

Every candidate reduces premature ends, but none holds fewer than 5% of the
final ends, so none is eligible. All answered in time: median readiness 56 ms
(L, the speculative ASR call), 67 ms (A), 98 ms (B, ASR plus the LLM call while
the speculative reply streamed on the same server) and 88 ms (stage 2), with at
most 0.3% of answers later than the 160 ms window.

## Why the false-hold rates are high

- **Unfinished clip ends.** The development clips are podcast segments whose
  ends are diarization boundaries. Of the 56 plain clips whose final end the
  lexical rule held, most end in a word that cannot end a Persian sentence
  («که» 9, «این» 5, «رو» 4, «مثلا» 4, «یعنی» 4, «یه» 4, …): the speaker had not
  finished. Part of every candidate's false-hold rate is therefore correct
  holding; the measure is conservative.
- **Smart Turn zero-shot does not transfer to Persian.** On held-out Persian
  examples its balanced accuracy is 0.53 (0.61 on the corrected set), near chance.
- **Persian training data with natural turn ends is scarce here.** Stage 2
  mined the user channel of the training-split conversation pairs (diarized
  windows of other sessions were exhausted by the exclusions). In 90% of the
  pairs the speaker was still voicing where the turn was cut (the other speaker
  took over), so only 405 pairs gave a natural end: 738 training and 72
  validation examples. Fine-tuning raised validation accuracy from 0.61 to
  0.79, but the model still holds most final ends of the development clips.

## Stage 2 and amendment 1

The first stage-2 model (9,308 training examples, validation accuracy 0.76)
is invalid: its *complete* examples ended at the hard cut of the pair audio
while the speaker was still voicing (last 60 ms at −0.8 dB relative to the
clip, against −13.0 dB for the *incomplete* examples cut at a pause), so it
learned the cut. On the development trials it held 92–97% of final ends.
Amendment 1, committed before the corrected run, cuts both classes where a
pause begins, as the live gate sees audio, and keeps a turn end only where the
speaker stopped naturally. The corrected model is the one in the table.

The conversion of the public Smart Turn v3.2 checkpoint into its training
model is exact (maximum difference 1.2e-7 against the published ONNX file);
the fine-tuned model is exported to ONNX and matches PyTorch (difference
1.2e-7). Training used the model class, loss and optimizer settings of the
public training code (learning rate 5e-5, 4 epochs, warm-up 0.2, weight decay
0.01) with batch 64.

## What stays

- The deployed endpointer is unchanged (256 ms confirmation, 96 ms speculative
  start); premature endpointing on mid-sentence pauses remains a reported
  failure mode (15/201 plain and 7/80 simulated-elderly utterances split in
  loopback v2).
- The gate mechanism (`Endpointer.hold`, `runtime/turn_end.py`, serve-live
  `--turn-end-gate`) stays in the code, off by default, with tests.
- A Persian semantic end-of-turn model needs labelled turn ends with natural
  endings (for example read or dialogue recordings segmented by people), not
  diarization boundaries of podcasts.

## Claim boundary

Offline replay on development audio only; automatic labels (diarization and a
VAD); simulated elderly speech. The loopback test panels were not used.
