# Moshi checkpoint-selection and held-out evaluation protocol

Frozen before the full H100 run: 2026-08-24.

## Selection rule

The training configuration evaluates every 250 steps and saves every 500 steps.
All 16 checkpoint-aligned candidates from steps 500 through 8,000 are retained.

The primary adapter is the saved checkpoint with the minimum finite mean
validation `eval_loss` in `checkpoints/moshi_fa/metrics.eval.jsonl`. If two
candidates have exactly equal loss, the earlier step wins. Selection fails
closed if step 8,000 is absent, any expected saved candidate is absent, or its
adapter/config cannot be hashed.

Run:

```bash
.venv/bin/python scripts/select_moshi_checkpoint.py
```

The selector records every candidate, loss, adapter/config size and SHA-256, and
the selected step in `results/moshi_checkpoint_selection.json`.

Then validate the selected artifact through the isolated pinned Moshi
environment:

```bash
.venv-moshi/bin/python scripts/validate_moshi_adapter.py --runtime-device cuda
```

Validation fails closed on stale hashes/config, a missing or unexpected
trainable key, any shape/dtype mismatch, non-finite values, or official-loader
failure. It records the exact adapter/config/selection hashes in
`results/moshi_adapter_validation.json`.

Training loss, the group-disjoint test split, subjective listening, latency,
and a favorite generated example must not influence checkpoint selection.

## One-time held-out evaluation after selection

After the selected adapter loads with no missing/unexpected trainable keys:

Run exactly once after selection and adapter validation:

```bash
.venv-moshi/bin/python scripts/evaluate_moshi_adapter.py
```

1. Freeze its adapter/config hashes.
2. Run adapter-on, adapter-off, and deterministically perturbed-adapter
   conditions on identical group-disjoint test inputs and seeds.
3. Verify target sensitivity by changing only the assistant target text/audio
   and confirming the corresponding text/audio-token losses change.
4. Report text and audio loss separately, Persian script/drift/failure rates,
   silence/loop/codec-collapse checks, and content-controlled ASR CER/WER.
5. Compare the adapted model, unadapted pinned Moshika, and the labelled
   NeMo→Qwen/rules→Piper cascade on the same eligible cases.
6. Keep the automatic sample separate from human listening: no perceptual,
   pronunciation, naturalness, or interruption-success claim passes without an
   actual reviewer/participant.

The test set is used once for final reporting. If the selected adapter fails a
predeclared safety/validity check, report the failure; do not select a different
checkpoint from the test result. Any new run must use a new, explicitly
versioned protocol and validation split decision.
