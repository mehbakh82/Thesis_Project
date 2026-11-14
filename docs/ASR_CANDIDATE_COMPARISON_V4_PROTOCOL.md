# ASR candidate comparison v4: protocol

Status: frozen on 2026-11-14, before any v4 prediction was generated and before
any v4 model was downloaded.

## Why a v4

On 2026-11-14 the user asked whether the deployed recognizer (NeMo v4 CTC) is
really the best available, whether NeMo v6 should replace it, and whether the
344-clip PSRB panel is large enough to separate them. A search of the public
leaderboards and Persian model releases on the same day found:

1. **A public Persian streaming recognizer missed by v3:**
   `mehdi-hf/nemotron-asr-streaming-farsi` (NVIDIA Nemotron 3.5 ASR streaming
   0.6B fine-tuned on 1,181 h of openly licensed Persian; last modified
   2026-10-03, before v3 was frozen). Its card reports 8.8% WER on FLEURS fa
   and 26.0% on a conversational test, in true streaming. v3's candidate search
   did not find it.
2. **The first-ranked model of the Open Persian ASR Leaderboard,**
   `C1Tech/whisper_small_persian`, was not tested; v3 tested only its base
   sibling (23.3% CER on PSRB). The repository is gated with manual approval.
3. **The user's own NeMo v8 checkpoint** (quality-aware pseudolabel fine-tune
   initialized from the v6 export) was never scored. Its best validation
   checkpoint (epoch 0, val WER 0.2511) is lower than v6's best (0.2520).
4. **Panel size.** On PSRB, v6 was 0.71 CER points worse than v4 (paired clip
   bootstrap 95% interval +0.40 to +1.02). A second human-reference panel of
   different speech tests whether the ordering holds.

Not included, with the reason:

- `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (first among open weights on
  the Artificial Analysis board): Persian is not in its 40-language support
  matrix; offline only; needs an 80 GB GPU.
- The leading entries of the Hugging Face Open ASR Leaderboard and the
  Artificial Analysis board: they are ranked on English (plus a few European
  languages), and the top closed APIs are cloud services that cannot run in the
  local 12 to 24 GB system.

## Already computed before this protocol (reported as post hoc)

On 2026-11-14, before this protocol was written, the stored v2 predictions of
NeMo v4 and v6 were compared clip by clip on PSRB: v4 has fewer character
errors on 111 clips, v6 on 71, 162 ties (exact two-sided sign test p = 0.004);
v4 is better or equal in every acoustic condition and source type. This is
reported as a post-hoc robustness check of v2, not as a v4 result.

## Panels

- **PSRB-human (primary, unchanged):** all 344 clips, panel SHA-256
  `bc80e48ead162f48323617389852b6a01a90c54eada6b11226696c78283d2e13`.
- **FLEURS fa_ir test (secondary, new):** every clip of `google/fleurs`
  config `fa_ir`, split `test`, at dataset revision
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (CC BY 4.0), reference =
  `raw_transcription`. Read Wikipedia sentences by native speakers, human
  transcripts. The panel file and its SHA-256 are written before any
  prediction on it. FLEURS may be in the training data of public checkpoints
  (the Nemotron card states it is not in its own).

Scoring is unchanged: `fa-asr-score-v1` through
`scripts/evaluate_asr_candidates_v2.py` (micro CER primary, WER, strata,
10,000-draw clip bootstrap, paired differences against `nemo_v4_ctc`).

## Arms

| Arm | Checkpoint (pinned) | Decoding | Runtime |
|---|---|---|---|
| `nemotron_fa_la1120` | `mehdi-hf/nemotron-asr-streaming-farsi` @ `b70aff58…` (safetensors) | 13 look-ahead tokens (1.12 s), greedy RNN-T, `fa-IR` prompt | transformers ≥ 5.18 |
| `nemotron_fa_la320` | same | 3 look-ahead tokens (0.32 s, the model default) | transformers ≥ 5.18 |
| `nemo_v8_ep00_ctc` | v8 quality-aware `epoch=00-step=15639-val_wer_ctc=0.2511.ckpt`, exported exactly as v6 was (`export_v6_for_v8.py` method: strict state-dict load into the same architecture, tensor-equality check) | greedy CTC | NeMo 2.7.3 |
| `c1tech_whisper_small_fa` | `C1Tech/whisper_small_persian` @ `685a77c4…` | language fa, beam 1 | transformers |
| `nemo_v4_ctc_rerun` | the deployed NeMo v4 | greedy CTC | NeMo 2.7.3 |

`c1tech_whisper_small_fa` runs only if access to the gated repository is
granted before the other v4 predictions are scored; otherwise it is reported as
not run.

The Nemotron arms transcribe each clip in one call (the card's offline usage),
not chunk by chunk; with a cache-aware encoder the look-ahead setting fixes the
attention context, but small differences from the card's chunked results are
possible and are not investigated.

On FLEURS, the new arms are scored together with `nemo_v4_ctc`, `nemo_v6_ctc`,
`shenava_koochik` and `seamless_v5ft_direct` (accuracy reference), with the
same checkpoints and decoding as in v2 and v3.

Predictions come from `scripts/asr_predict.py` one clip at a time after one
warm-up clip; errors and empty outputs count as full deletions.

## Selection rule

1. **Latency eligibility (v2 rule):** median per-utterance latency on PSRB clips
   of at most 8 s no higher than 150 ms, measured in this session; the
   `nemo_v4_ctc_rerun` arm gives the same-session reference.
2. **Replacement of NeMo v4:** an eligible arm replaces it only if (a) it has
   the lowest PSRB micro CER among the eligible arms of v2, v3 and v4, (b) its
   paired PSRB CER difference against `nemo_v4_ctc` has a 95% upper bound below
   0, (c) its PSRB senior-stratum CER is not higher than NeMo v4's, and (d) on
   FLEURS its paired CER difference against `nemo_v4_ctc` is not significantly
   positive (95% lower bound at most 0).
3. **Memory and live confirmation:** unchanged from v3 (rules 3 and 4 of
   `docs/ASR_CANDIDATE_COMPARISON_V3_PROTOCOL.md`), including one formal
   loopback run before a replacement becomes the deployed default.
4. **NeMo v4 against v6 (descriptive):** the FLEURS paired difference is
   reported with its interval. If it favours v6 significantly, the thesis
   states that the two panels disagree; it does not change rule 2.

## Claim boundary

Two public human-reference panels (53 minutes of mostly spontaneous speech and
read FLEURS sentences); per-utterance latency on the shared H100 host; public
checkpoints may have seen PSRB-like or FLEURS data. "Best" means best among the
arms tested on these panels on this date.
