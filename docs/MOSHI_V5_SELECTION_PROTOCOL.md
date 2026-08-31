# Moshi v5 reduced semantic-codebook-weight checkpoint protocol

Frozen: 2026-08-31, after v4 selection failed closed and before any v5 probe,
optimizer step, generation, or access to the frozen final-test rows.

## Evidence and single-factor hypothesis

V4 used rank-128 LoRA, selectively trained exactly `text_emb.weight` and
`depformer_text_emb.weight`, froze all 23 audio-codebook embeddings, and used
`first_codebook_weight_multiplier: 100.0`. It passed 1/9 runtime-panel rows at
steps 100 and 400 and 0/9 elsewhere. Speech-producing rows declined from 9/9 at
step 100 to 5/9 at step 500 even while complete validation loss improved from
2.082125 to 1.879061. V4 selected no checkpoint and never accessed the final
test. Canonical evidence is `docs/MOSHI_V4_RESULT.md` and
`results/hardware/moshi_h100_v4_training.json`.

V5 changes exactly one training-profile value from v4:

- `first_codebook_weight_multiplier`: `100.0` to `10.0`

This multiplier weights the first, semantic audio codebook inside the summed
audio loss. The falsifiable hypothesis is that 100-fold pressure dominates the
adaptation objective and contributes to autoregressive speech collapse;
retaining a 10-fold emphasis may preserve speech while leaving greater relative
capacity for Persian text and the remaining acoustic codebooks. A value of 10
is predeclared as a conservative intermediate value, not chosen after observing
v5 outputs.

Everything else is held fixed: base model, data and group split, 100-second
context, effective batch of four microbatches, rank-128 LoRA, the exact two
trainable text embeddings, all frozen audio embeddings, text-padding weight,
optimizer, learning rate and schedule, seed, 500-step horizon, candidate steps,
runtime panel, selection rule, and eligibility thresholds. Training starts from
the pinned base model, never from a v2-v4 adapter.

## Bounded candidate set and probe

The complete candidate set is `[100, 200, 300, 400, 500]`, with all five
checkpoints retained through certification. Before training,
`configs/moshi_h100_v5_probe.yaml` must complete exactly one optimizer step and
one CPU-offloaded checkpoint at the exact v5 shape. It must demonstrate finite
loss, hardware headroom, and the exact 676-tensor schema: 674 LoRA tensors plus
the two declared text embeddings and no audio embedding. The probe is not a
selection candidate.

## Selection gates

Selection uses only the unchanged 202-chunk validation manifest. The nine
complete-prompt panel indices remain `[0, 11, 33, 51, 55, 74, 85, 87, 106]`.
A candidate is eligible only if:

1. hashes, saved config, exact adapter schema, BF16 dtypes, and finite values pass;
2. text, audio, and total loss are finite over the identical complete validation scope;
3. the pinned official server and attested client evaluate all nine rows;
4. every row has finite decoded audio, RMS at least 0.001, nonempty text, and
   Persian-script fraction at least 0.5; and
5. all nine rows pass without any fallback model or post-processing cascade.

Among eligible candidates, choose minimum complete-scope total validation loss;
an exact tie chooses the earlier step. No threshold, panel member, candidate,
or rule may change after outputs are observed. If none is eligible, v5 fails
closed.

## Final-test firewall

The untouched 738-row, 14-session final test remains sealed. No row, objective,
generation, or metric may be read before an eligible selection and exact
artifact hashes are committed from a clean worktree. If and only if v5 selects
an eligible checkpoint, validate that exact adapter, commit the selection
evidence, and access the final test once. Test results can never revise the
selection. A null selection permanently prevents the v5 final stage.

## Storage and stopping rules

Preflight reserves space for one probe, five retained candidates, one transient
save, and 2 GiB safety headroom. Stop and record failure on non-finite loss,
schema mismatch, ENOSPC, OOM, missing selective-scope journal marker, missing
checkpoint, or any premature final-test access. A clean 500-step run remains a
scientific negative unless every frozen eligibility gate passes.
