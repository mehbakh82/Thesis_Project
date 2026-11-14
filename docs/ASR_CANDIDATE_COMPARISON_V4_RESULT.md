# ASR candidate comparison v4: result

Protocol: `docs/ASR_CANDIDATE_COMPARISON_V4_PROTOCOL.md` (frozen 2026-11-14,
commit `0c23980`, before any v4 prediction). Receipt:
`results/eval/asr_candidates_v4.json` (all v2, v3 and v4 arms on PSRB and the
v4 FLEURS arms, scored by the unchanged v2 scorer). **Outcome: NeMo v4 stays
the deployed recognizer.** No new arm is both fast enough and significantly
more accurate on the conversational panel.

## Results

PSRB-human: 344 clips, 53 minutes, mostly spontaneous speech, human
references. FLEURS fa_ir test: 871 read sentences, 3.7 hours, human
references. CER differences are paired against NeMo v4 with 95% clip-bootstrap
intervals (percentage points; negative = fewer errors than v4).

| Arm | PSRB CER | vs v4 | FLEURS CER | vs v4 | PSRB WER | Median latency, PSRB clips ≤ 8 s |
|---|---:|---|---:|---|---:|---:|
| Seamless v5ft (accuracy reference) | 6.92% | −2.52 (−3.23 to −1.87) | 3.91% | −1.37 (−1.56 to −1.18) | 21.0% | 551 ms |
| Shenava Koochik (v3) | 8.95% | −0.50 (−1.07 to +0.04) | 4.84% | −0.43 (−0.57 to −0.30) | 35.3% | 61 ms |
| **Nemotron Persian, 1.12 s look-ahead** | 9.01% | −0.43 (−1.00 to +0.15) | 5.05% | −0.22 (−0.39 to −0.06) | 24.7% | 215 ms |
| **NeMo v4 (deployed)** | 9.44% | | 5.28% | | 24.0% | 55 ms (57 ms rerun) |
| **Nemotron Persian, 0.32 s look-ahead** | 9.80% | +0.35 (−0.31 to +1.04) | 5.14% | −0.14 (−0.31 to +0.04) | 26.0% | 167 ms |
| NeMo v6 (v2) | 10.15% | +0.71 (+0.40 to +1.02) | 5.34% | +0.07 (−0.02 to +0.16) | 24.3% | 58 ms |
| **NeMo v8, epoch 0** | 10.18% | +0.73 (+0.42 to +1.05) | 5.39% | +0.12 (+0.03 to +0.21) | 24.5% | 61 ms |

New v4 arms in bold. The same-session rerun of NeMo v4 reproduced all 344 v2
transcripts exactly.

## Decision (pre-registered rule)

- **Latency-eligible arms (≤ 150 ms):** Shenava, the NeMo arms (v4, v6, v8),
  omniASR-CTC-1B, Rade, NVIDIA FastConformer and NeuraSpeech. Both Nemotron
  arms are slower than the budget in this runtime (Transformers 5.23, greedy
  RNN-T, one call per clip).
- **Lowest PSRB CER among them:** Shenava Koochik, as in v3. Its PSRB gain
  over v4 is not significant (interval includes zero), so rule 2(b) fails and
  NeMo v4 stays. On FLEURS, Shenava is significantly better than v4 and its
  senior CER on PSRB is not higher; its word error rate is much higher on both
  panels (+11.4 and +9.6 points), mostly from word-spacing conventions.
- **Nemotron at 1.12 s look-ahead** would not replace v4 even with a faster
  runtime: its PSRB gain (−0.43) is not significant, although its FLEURS gain
  (−0.22) is. A NeMo-runtime latency check was therefore not needed.
- **NeMo v8** is significantly worse than v4 on both panels.

## NeMo v4 against v6

On PSRB, v6 is significantly worse than v4 (+0.71 points). A post-hoc clip-by-
clip check of the stored v2 predictions, done before this protocol was
written, agrees: v4 has fewer character errors on 111 clips and v6 on 71 (162
ties; exact sign test p = 0.004), and v4 is better or equal in every acoustic
condition and source type. On FLEURS the two are tied (+0.07, interval
−0.02 to +0.16). The two panels therefore do not disagree: v6 is never better
than v4, and on conversational speech it is worse. The 344-clip panel was
large enough to detect this 0.7-point difference.

## Not included

- `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`: Persian is not among its 40
  languages; offline only; 80 GB GPU.
- `C1Tech/whisper_small_persian` (first on the Open Persian ASR Leaderboard):
  the repository is gated with manual approval, and the user decided not to
  request access. Its base sibling scored 23.3% CER on PSRB in v3.

## Deviations

- The FLEURS audio and references came from the dataset's
  `refs/convert/parquet` branch (commit `168de341…`, file SHA-256
  `e8982e2e…`), the Hub's parquet conversion of the revision named in the
  protocol. The panel has 871 clips (the model card that motivated the panel
  counted 852); every clip is used, as the protocol requires.
- Nemotron ran in a separate environment with Transformers 5.23 and the
  project's torch 2.12 (the card asks for Transformers ≥ 5.18).

## Claim boundary

Two public human-reference panels (53 minutes of mostly spontaneous speech and
3.7 hours of read sentences); latency per utterance on the shared H100 host;
public checkpoints may have seen PSRB-like or FLEURS data. "Best" means best
among the 24 arms tested on these panels on 2026-11-14.
