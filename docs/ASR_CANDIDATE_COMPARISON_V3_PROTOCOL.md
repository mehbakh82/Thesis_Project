# ASR candidate comparison v3 — protocol

Status: frozen on 2026-10-28, before any v3 prediction was generated.

## Why a v3

The v2 comparison (`docs/ASR_CANDIDATE_COMPARISON_V2_PROTOCOL.md`) left out
every Persian checkpoint that was not already in the thesis pipeline. One of
them, Rade-ASR-CTC-3B-fa, was left out only because its 6.16 GB download did
not complete under the earlier disk and transfer limits
(`results/hardware/rade_asr_runtime_attempt.json`). The others were screened
only in the user's separate noisy-speech study, never on the thesis's human-
reference panel. v3 closes both gaps with the v2 panel, scorer and rule.

## Panel and scoring (unchanged from v2)

PSRB-human, all 344 clips (panel SHA-256
`bc80e48ead162f48323617389852b6a01a90c54eada6b11226696c78283d2e13`), scored
with `fa-asr-score-v1` by `scripts/evaluate_asr_candidates_v2.py`: micro CER
(primary) and WER, the senior, acoustic and spontaneity strata, 10,000-draw
clip bootstrap intervals and paired differences against `nemo_v4_ctc`. The
panel was already used by v2; no threshold or decoding setting is tuned on it
here.

## Arms

| Arm | Checkpoint (pinned) | Decoding | Runtime |
|---|---|---|---|
| `rade_ctc_3b` | `RadeAI/Rade-ASR-CTC-3B-fa` @ `70129ddc…`, `model_fp16.pt` (LFS SHA-256 `3f3c1144…`) | greedy CTC, fp16, `pes_Arab` | official `omnilingual-asr` 0.2.0 / fairseq2 0.6 pipeline, as on the model card |
| `nvidia_fa_fastconformer_ctc` | `nvidia/stt_fa_fastconformer_hybrid_large` | greedy CTC | NeMo |
| `neuraspeech_900h` | `Neurai/NeuraSpeech_900h` | saved RNNT greedy decoder | NeMo |
| `buzzasr_persian` | `BuzzASR/persian` (Whisper large-v3 fine-tune) | author's published beam-1 settings | transformers |
| `whisper_persian_v4` | local `whisper-persian-v4` (Whisper large-v3 fine-tune) | language fa, beam 1 | transformers |
| `whisper_large_v3_turbo_fa` | local Whisper large-v3-turbo Persian fine-tune | language fa, beam 1 | transformers |
| `c1tech_whisper_base_fa` | `C1Tech/whisper_base_persian` | language fa, beam 1 | transformers |
| `shenava_koochik` | `Reza2kn/Shenava-Koochik-v1.0` (the NeMo checkpoint behind the sherpa-onnx export screened in 2026-09) | greedy CTC | NeMo |
| `nemo_v4_ctc_rerun` | the deployed NeMo v4 checkpoint | greedy CTC | NeMo |

Not included: Vosk small Persian 0.42 (a CPU Kaldi model of 2021 that was the
second-worst arm of the user's screen) and the autoregressive omniASR-LLM-1B
(its CTC sibling scored 14.0% CER in v2, and Rade is the Persian fine-tune of
the larger CTC-3B).

`nemo_v4_ctc_rerun` repeats the incumbent in the same session, so that latency
is compared under the same GPU load. Its transcripts must equal the v2
incumbent's; paired accuracy differences use the v2 `nemo_v4_ctc` predictions.
Model hashes are recorded in each arm's metadata. Predictions come from
`scripts/asr_predict.py` one clip at a time after one warm-up clip; errors and
empty outputs count as full deletions.

## Selection rule

1. **Latency eligibility (v2 rule):** median per-utterance latency on clips of
   at most 8 s no higher than 150 ms, measured in this session.
2. **Replacement of the deployed ASR:** an eligible arm replaces NeMo v4 only
   if it has the lowest micro CER among the eligible arms of v2 and v3, its
   paired CER difference against `nemo_v4_ctc` has a 95% upper bound below 0,
   and its senior-stratum CER is not higher than NeMo v4's.
3. **Memory:** the deployed stack must still fit the proposal's 24 GB ceiling
   (ASR process plus the responder process at most 22 GiB together, leaving
   room for the CUDA contexts). Whether it also fits the 12 GiB profile is
   reported.
4. **Confirmation in the live system:** a replacement becomes the deployed
   default only after one formal loopback run with the v2 harness, panels, host
   and scorer (`docs/LIVE_LOOPBACK_V2_PROTOCOL.md`) keeps the R3 criteria
   (plain mean and p90 ≤ 500 ms, simulated-elderly p90 ≤ 500 ms) and an
   interruption accuracy of at least 0.80 in the browser. Otherwise NeMo v4
   stays and the winner is reported as the more accurate offline recognizer.

The lowest-CER arm regardless of latency stays the accuracy reference (v2:
Seamless v5ft, 6.92%) unless a v3 arm is lower.

## Claim boundary

One human-reference public panel of 53 minutes; per-utterance latency on the
shared H100 host; PSRB may overlap the training data of public checkpoints.
