# Moshi v7: romanized, word-aligned inner monologue

Status: frozen before the first v7 optimizer step.

## Why a v7 after the v6 stop rule

The v6 decision rule said a negative 32-row diagnostic should stop further
Moshika LoRA work *under the tested hypothesis* (insufficient text-parameter
scope). A capacity audit performed afterwards measured a different, never-tested
bottleneck in the training targets themselves
(`results/moshi_tokenizer_audit.json`, `scripts/audit_moshi_tokenizer.py`):

| Inner-monologue text | Tokens / word | Tokens / 12.5 Hz frame | Turns over frame budget |
|---|---:|---:|---:|
| Persian script (v1–v6.2) | 6.59 (55% byte fallback; 8/33 letters have a piece) | 1.64 | 98.5% |
| ASCII romanized Piper phonemes (v7) | 2.67 | 0.67 | 0.3% |

Moshi emits at most one text token per frame. Every earlier version also used
one alignment span per whole assistant turn, while the pinned interleaver
(`keep_and_shift=false`) expects word timestamps: with one span, the text
stream drifts ever further behind the speech; with word timestamps but
Persian script, untransmitted tokens of a word are dropped when the next word
starts. v7 tests whether removing this mismatch makes the unchanged v6 recipe
produce correct autoregressive output.

## Intervention (exactly two changes versus v6)

1. **Text representation:** each assistant word is the deterministic ASCII
   romanization (`src/thesis_s2s/data/romanize.py`, version
   `piper-espeak-fa-ascii-v1`) of the espeak-ng phonemes that the pinned Mana
   Piper voice actually spoke when rendering the assistant channel.
2. **Word-level timing:** each word's start/end comes from Piper's own
   phoneme-to-sample alignment of a re-synthesis that must reproduce the stored
   assistant channel (equal length, correlation ≥ 0.999). Audio is never
   rewritten (`scripts/prepare_moshi_v7.py`).

Everything else is v6: the same 32 train-only rows and order, rank-64 LoRA plus
full text embeddings and untied text output projection, audio loss × 0.1,
12-second duration, effective batch 4, 200 steps, learning rate, seed, the
Gloo backend, and checkpoints at steps 50/100/150/200
(`configs/moshi_h100_v7a_overfit.yaml`, `configs/moshi_v7a_overfit_policy.json`).

## Stage v7a: 32-row diagnostic and its positive criterion

The official Moshi server is exercised on the same deterministic nine-row panel
(indices 0, 3, 7, 11, 15, 19, 23, 27, 31) with the v6.1 decoding configuration
(`configs/moshi_v6_text_greedy_runtime.json`: greedy text, unchanged audio
sampling), so v6.1 is the controlled baseline (its passes were 0/9, 0/9, 1/9,
3/9). Each of the nine rows must:

1. stream the complete user turn;
2. return finite decoded audio with RMS at least `1e-3`;
3. return at least four normalized text characters, at least 80% of whose
   letters are ASCII `a–z` (the romanized analogue of v6's Persian-letter rule);
4. contain no Unicode replacement character; and
5. have normalized character error rate at most `0.75` against the
   equal-length prefix of that row's romanized training target.

The selected candidate's teacher-forced text loss must be at least 30% below
step 50 unless step 50 itself passes all nine rows. The earliest fully passing
candidate is selected. No threshold may change after outputs are seen.

**Predeclared secondary measurement (not a v7a gate):** the generated assistant
speech of every panel row is saved and transcribed by the project's selected
Persian ASR; its character error rate against the equal-length prefix of the
Persian target is reported as an audible-intelligibility indicator.

## Decision rule

- **v7a positive:** freeze v7b — the same two changes applied to the corrected
  v2–v5 training/validation split (5,681 / 131 rows) — and train from the base.
  v7b's validation gate is the frozen nine-row v2–v5 validation panel with the
  v7a row rules plus the secondary ASR intelligibility rule promoted to a gate
  (Persian round-trip CER ≤ 0.75). Only an eligible v7b checkpoint may open the
  sealed 738-row test, exactly once, for held-out loss versus the unadapted base
  and the same runtime rules on a predeclared test panel.
- **v7a negative:** the tokenizer/alignment hypothesis is falsified for this
  recipe; stop Moshika training and keep the direct-model claim negative.

The sealed v2–v5 test split remains untouched throughout v7a.

## Commands

```bash
PYTHONPATH=src <python with piper-tts+onnx> scripts/prepare_moshi_v7.py \
  --source data/processed/moshi_finetune_v6_overfit/train.jsonl \
  --out-dir data/processed/moshi_v7a_overfit --split train \
  --receipt results/moshi_v7a_overfit_data.json

env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 MOSHI_DISTRIBUTED_BACKEND=gloo \
  MOSHI_PERSIAN_TEXT_ADAPTATION=1 MOSHI_TEXT_EMBEDDINGS_ONLY=0 \
  MOSHI_AUDIO_LOSS_WEIGHT=0.1 \
  .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
  scripts/moshi_train_entry.py configs/moshi_h100_v7a_overfit.yaml

env CUDA_VISIBLE_DEVICES=0 .venv-moshi/bin/python scripts/reevaluate_moshi_v6_overfit.py \
  --training-config configs/moshi_h100_v7a_overfit.yaml \
  --run-dir checkpoints/moshi_v7a_overfit \
  --metrics-out checkpoints/moshi_v7a_overfit/metrics.reeval.jsonl \
  --report-out results/moshi_v7a_overfit_reevaluation.json \
  --experiment-label v7a_romanized_word_aligned

env CUDA_VISIBLE_DEVICES=0 .venv-moshi/bin/python scripts/evaluate_moshi_v7_overfit.py
```
