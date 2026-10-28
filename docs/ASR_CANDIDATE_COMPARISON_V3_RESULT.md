# ASR candidate comparison v3 — result

Protocol: `docs/ASR_CANDIDATE_COMPARISON_V3_PROTOCOL.md`. Receipt:
`results/eval/asr_candidates_v3_psrb_human.json` (all v2 and v3 arms, scored
together by the unchanged v2 scorer). **Outcome: NeMo v4 stays the deployed
recognizer.** The most accurate arm within the latency budget, Shenava Koochik,
is 0.5 points better, but the difference is not significant.

## Results (PSRB-human, 344 clips, human references)

| Arm | CER | 95% CI | Senior CER | Median latency, clips ≤ 8 s |
|---|---:|---|---:|---:|
| Seamless v5ft (v2, accuracy reference) | 6.92% | 5.8–8.1 | 10.4% | 551 ms (direct call) |
| **Shenava Koochik** | **8.95%** | 7.7–10.3 | 14.5% | 61 ms |
| NeMo v4 (deployed) | 9.44% | 8.0–11.0 | 15.6% | 55 ms (v2), 70 ms (rerun today) |
| BuzzASR Persian | 15.8% | — | 20.0% | 471 ms |
| Rade-ASR-CTC-3B-fa | 16.1% | 13.1–19.2 | 30.0% | 115 ms |
| NVIDIA Persian FastConformer | 19.2% | — | 26.4% | 66 ms |
| Whisper large-v3-turbo, Persian fine-tune | 20.2% | — | 22.3% | 411 ms |
| Whisper large-v3, Persian fine-tune v4 | 21.5% | — | 24.5% | 1,121 ms |
| C1Tech Whisper base Persian | 23.3% | — | 28.2% | 834 ms |
| NeuraSpeech 900h | 25.4% | — | 37.4% | 72 ms |

All intervals and per-stratum figures are in the receipt. The NeMo v4 rerun
reproduced all 344 v2 transcripts exactly; its latency was higher because the
shared GPU was busier.

## Decision (pre-registered rule)

- **Latency-eligible arms:** Shenava, the four NeMo arms, omniASR-CTC-1B,
  Rade, NVIDIA FastConformer and NeuraSpeech.
- **Lowest CER among them:** Shenava Koochik, 8.95% against NeMo v4's 9.44%.
  The paired difference is −0.50 points, with a 95% interval of −1.07 to
  +0.04, which includes zero.
- **Seniors:** Shenava's senior CER is not higher (14.5% against 15.6%).

The rule requires the interval to lie below zero, so Shenava does not replace
NeMo v4, and no loopback re-run is needed. Shenava's word error rate is much
higher than NeMo's (35.3% against 24.0%), mostly from word-spacing
conventions; CER is the pre-registered primary measure.

## Rade-ASR-CTC-3B-fa

Rade was the only Persian checkpoint left out earlier for disk and transfer
reasons. It ran through the official omnilingual-asr 0.2.0 pipeline exactly as
on its model card (fp16, greedy CTC, `pes_Arab`), with the verified checkpoint
(SHA-256 `3f3c1144…`), at 6.7 GiB of GPU memory.

On PSRB its CER is 16.1%. Most of its errors are deletions:

- 14.0% of reference characters are deleted, against 5.2% for NeMo v4, while
  its substitution rate is half of NeMo's (1.5% against 2.9%).
- The dropped speech is often at the start of a clip: 21 clips lose
  more than a fifth of their opening, and 30 of the 175 clips up to 8 s keep
  less than 70% of their content.
- Senior speakers suffer most (30.0% CER).

Its tokenizer writes «⁇» where Persian uses a half-space. The scorer strips it
as punctuation, so CER is the same with or without spaces (16.1%).

Its published scores (4.4% CER on FLEURS read speech) did not transfer to this
panel of mostly spontaneous, partly noisy speech. These findings come from the
official pipeline as published; we did not debug the model.

## Claim boundary

One public human-reference panel of 53 minutes; latency per utterance on the
shared H100 host; public checkpoints may have seen PSRB-like data.
