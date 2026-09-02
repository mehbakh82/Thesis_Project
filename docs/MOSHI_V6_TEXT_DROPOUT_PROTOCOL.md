# Moshi v6.2 scheduled text-input-dropout protocol

Status: frozen after the completed v6.1 negative and before any v6.2 probe or optimizer step.

## Scientific scope

This is a train-only capacity and exposure-bias diagnostic. It is not validation,
held-out-test, or deployment evidence. The final test is not an input and must not
be opened. The same mechanically filtered, not human-verified 32 rows and the same
nine-row panel used by v6/v6.1 remain fixed.

The parent evidence is:

- v6 runtime SHA-256 `fc3c394a76e8866baf40fe649a0ec5c78d0b857aa1307d3c2cbd9caf16961d6b`,
  an infrastructure-valid negative;
- v6.1 runtime SHA-256 `220e487e0a62e232c4ba479e87abdace4d3df4515d2a94c647371a882a4bc748`,
  also negative, with deterministic text decoding improving step 200 from
  `1/9` to `3/9`.

## Hypothesis and single controlled change

v6 reduced clean teacher-forced text loss by 96.3% but failed autoregressively.
v6.1 showed that decoding randomness explains only part of that gap. The remaining
hypothesis is exposure bias: training always conditions on the true previous text
while runtime conditions on generated text.

v6.2 keeps v6's data, model, trainable parameters, audio weight, optimizer,
learning-rate schedule, seed, 200 steps, four microbatches, and checkpoint cadence.
Only training inputs change:

- lexical text tokens (IDs greater than 3) are independently replaced with the
  existing text-padding token (ID 3);
- probability increases linearly from 0.25 on training forward 1 to 0.75 on
  training forward 800 (200 steps times four microbatches);
- RNG seed is 20260902;
- IDs 0, 1, 2, and 3, every audio code, and the original loss targets are unchanged;
- evaluation is always clean and never consumes corrupted inputs.

A one-step/four-microbatch no-checkpoint probe must prove the hook executes
through all four forwards, changes eligible tokens, leaves targets/evaluation
clean, has finite loss, and stays within the
previously validated memory profile.

## Training and candidates

The exact config is `configs/moshi_h100_v6_text_dropout.yaml`. Checkpoints are
50, 100, 150, and 200. Corrected clean loss reevaluation must cover all 32 rows at
every candidate. Official runtime uses no-fuse LoRA and the already justified
v6.1 decoding policy: audio top-k 250 and text top-k 1.

## Positive criterion

A candidate is positive only if every one of the same nine rows passes every
unchanged gate: complete user turn, valid official-server exercise, finite audible
audio with RMS at least `1e-3`, at least four normalized text characters,
Persian-letter fraction at least `0.80`, no Unicode replacement character, and
target-prefix CER at most `0.75`. The earliest all-nine candidate is selected
only if its clean fixed-scope text loss is at least 30% below step 50, unless step
50 itself passes.

No threshold may be relaxed after results are visible.

## Decision

- Positive: retain the selected v6.2 adapter as train-only proof that scheduled
  input corruption repairs the observed exposure failure; generalization remains
  unproven.
- Negative: preserve the result. Do not claim v6 positive. Further direct-model
  work requires a separately frozen intervention (such as audio-stream corruption
  or a different model), while the Persian cascade remains the deployable path.

## Exact execution sequence

Run the write-once preflight from a clean commit:

```bash
.venv/bin/python scripts/preflight_moshi_v6_text_dropout.py
```

Run the no-checkpoint probe and its recorder with the same shell environment:

```bash
env CUDA_VISIBLE_DEVICES=0 MOSHI_DISTRIBUTED_BACKEND=gloo \
  MOSHI_PERSIAN_TEXT_ADAPTATION=1 MOSHI_TEXT_EMBEDDINGS_ONLY=0 \
  MOSHI_AUDIO_LOSS_WEIGHT=0.1 MOSHI_TEXT_INPUT_DROPOUT_START=0.25 \
  MOSHI_TEXT_INPUT_DROPOUT_END=0.75 MOSHI_TEXT_INPUT_DROPOUT_FORWARDS=4 \
  MOSHI_TEXT_INPUT_DROPOUT_SEED=20260902 \
  MOSHI_TEXT_INPUT_DROPOUT_AUDIT=checkpoints/moshi_v6_text_dropout_probe/input_dropout_audit.jsonl \
  .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
  scripts/moshi_train_entry.py configs/moshi_h100_v6_text_dropout_probe.yaml

env MOSHI_DISTRIBUTED_BACKEND=gloo MOSHI_PERSIAN_TEXT_ADAPTATION=1 \
  MOSHI_TEXT_EMBEDDINGS_ONLY=0 MOSHI_AUDIO_LOSS_WEIGHT=0.1 \
  MOSHI_TEXT_INPUT_DROPOUT_START=0.25 MOSHI_TEXT_INPUT_DROPOUT_END=0.75 \
  MOSHI_TEXT_INPUT_DROPOUT_FORWARDS=4 MOSHI_TEXT_INPUT_DROPOUT_SEED=20260902 \
  MOSHI_TEXT_INPUT_DROPOUT_AUDIT=checkpoints/moshi_v6_text_dropout_probe/input_dropout_audit.jsonl \
  .venv/bin/python scripts/record_moshi_v6_text_dropout_probe.py
```

After committing the passed preflight/probe evidence, run the full diagnostic and
attest it in the same environment, changing only forwards/audit path:

```bash
env CUDA_VISIBLE_DEVICES=0 MOSHI_DISTRIBUTED_BACKEND=gloo \
  MOSHI_PERSIAN_TEXT_ADAPTATION=1 MOSHI_TEXT_EMBEDDINGS_ONLY=0 \
  MOSHI_AUDIO_LOSS_WEIGHT=0.1 MOSHI_TEXT_INPUT_DROPOUT_START=0.25 \
  MOSHI_TEXT_INPUT_DROPOUT_END=0.75 MOSHI_TEXT_INPUT_DROPOUT_FORWARDS=800 \
  MOSHI_TEXT_INPUT_DROPOUT_SEED=20260902 \
  MOSHI_TEXT_INPUT_DROPOUT_AUDIT=checkpoints/moshi_v6_text_dropout/input_dropout_audit.jsonl \
  .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
  scripts/moshi_train_entry.py configs/moshi_h100_v6_text_dropout.yaml

env MOSHI_DISTRIBUTED_BACKEND=gloo MOSHI_PERSIAN_TEXT_ADAPTATION=1 \
  MOSHI_TEXT_EMBEDDINGS_ONLY=0 MOSHI_AUDIO_LOSS_WEIGHT=0.1 \
  MOSHI_TEXT_INPUT_DROPOUT_START=0.25 MOSHI_TEXT_INPUT_DROPOUT_END=0.75 \
  MOSHI_TEXT_INPUT_DROPOUT_FORWARDS=800 MOSHI_TEXT_INPUT_DROPOUT_SEED=20260902 \
  MOSHI_TEXT_INPUT_DROPOUT_AUDIT=checkpoints/moshi_v6_text_dropout/input_dropout_audit.jsonl \
  .venv/bin/python scripts/record_moshi_v6_text_dropout_training.py

env CUDA_VISIBLE_DEVICES=0 .venv-moshi/bin/python \
  scripts/reevaluate_moshi_v6_overfit.py \
  --training-config configs/moshi_h100_v6_text_dropout.yaml \
  --run-dir checkpoints/moshi_v6_text_dropout \
  --metrics-out checkpoints/moshi_v6_text_dropout/metrics.reeval.jsonl \
  --report-out results/moshi_v6_text_dropout_reevaluation.json \
  --experiment-label v6_text_dropout

env CUDA_VISIBLE_DEVICES=0 .venv-moshi/bin/python \
  scripts/evaluate_moshi_v6_overfit.py --text-dropout-followup
```
