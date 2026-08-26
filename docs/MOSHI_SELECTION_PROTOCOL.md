# Moshi checkpoint-selection and held-out evaluation protocol

Frozen before the full H100 run: 2026-08-24.

## Selection rule

The training configuration evaluates every 250 steps and saves every 500 steps.
All 16 checkpoint-aligned candidates from steps 500 through 8,000 are retained.

The frozen criterion was the saved checkpoint with the minimum finite mean
validation `eval_loss`; exact ties choose the earlier step. Selection fails
closed if training is incomplete, any expected saved candidate is absent, or
its adapter/config cannot be hashed. The original plan named the pinned
trainer's `metrics.eval.jsonl` as the input. The post-run correction below
records why that file was rejected before selection while the criterion,
candidate set, and held-out isolation remained unchanged.

## Post-run validation-execution correction (2026-08-26)

The full run exposed a bug in the pinned upstream evaluation loop. It created
one finite validation iterator and reused it at every evaluation. Each call
processed 40 chunks but fetched and discarded a 41st at the break. The 682
chunks were therefore exhausted at step 4,250, and evaluation divided by zero
from step 4,500. Earlier values also came from different disjoint subsets and
were not comparable. Adapter tensors at steps 4,000, 4,500, and 8,000 were all
finite; fixed-scope step-4,500 evaluation was also finite, ruling out model
divergence.

Before selection and without consulting the test split, the project therefore
reevaluated all 16 predeclared saved candidates on the same complete, ordered
682-chunk validation set:

```bash
.venv-moshi/bin/python scripts/reevaluate_moshi_checkpoints.py
```

The report `results/moshi_validation_reevaluation.json` records the source bug,
raw and corrected metric hashes, exact common scope, every adapter/config hash,
and all corrected text/audio/total losses. This changed the invalid execution
input—not the frozen minimum-loss rule or candidate set. Future launches reset
the finite iterator on every periodic evaluation; final selection still uses
the complete fixed-scope reevaluation.

Run:

```bash
.venv/bin/python scripts/select_moshi_checkpoint.py
```

The selector accepts only a passed, hash-current fixed-scope reevaluation and
records every candidate, loss, validation scope, adapter/config size and
SHA-256, and the selected step in `results/moshi_checkpoint_selection.json`.

The corrected losses decreased from 1.649382 at step 500 to 1.402444 at step
8,000, so step 8,000 was selected. No test data or subjective output influenced
that decision.

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
Run:
