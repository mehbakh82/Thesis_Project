# Moshi v3 embedding-preserving checkpoint protocol

Frozen: 2026-08-29, after Moshi v2 selection failed closed and before any v3
optimizer step, v3 generation, or access to the frozen final-test rows.

## Motivation and controlled change

Moshi v2 completed all 2,000 steps and improved teacher-forced validation loss,
but no checkpoint passed all nine corrected complete-prompt official-runtime
gates. Step 400 was best at 2/9; early checkpoints spoke English and later
checkpoints increasingly became silent.

The pinned upstream 7B example uses rank-128 LoRA and leaves the embedding
matrices frozen. V2 instead used rank 64 with `ft_embed: true`. Its adapter
contained about 0.583 GiB of fully tuned embedding tensors and 0.361 GiB of
rank-64 LoRA tensors. V3 restores the upstream choice:

- `lora.rank: 128`
- `lora.ft_embed: false`

This is an embedding-preserving upstream-default correction, not a claim that
v2 proved a single-variable causal effect. The data, train/validation groups,
100-second duration, effective batch of four microbatches, loss weights,
optimizer, learning rate, seed, and all autoregressive output thresholds remain
unchanged.

The pinned upstream example is
`third_party/checkouts/moshi-finetune/example/moshi_7B.yaml`, SHA-256
`c574c424d1a078a89bcb7556978d8b4b8c160276aeebbda01ff77ca6f259ac48`.

## Bounded candidate set

V3 is a 500-step scientific screen because v2's strongest runtime behavior
occurred at step 400 and degraded afterward. The predeclared candidates are
steps `[100, 200, 300, 400, 500]`; every candidate must be retained. The
configuration is `configs/moshi_h100_v3.yaml`.

Before launch, an exact-shape one-step probe using
`configs/moshi_h100_v3_probe.yaml` must pass model/Mimi load, loss, backward,
optimizer, and CPU-offloaded adapter save. The probe is feasibility evidence,
not a selection candidate.

## Selection gates

Selection uses only the unchanged v2 validation manifest. The panel rule is now
predeclared before v3 training: retain rows whose complete user-channel PCM ends
within the five-second stream, then select nine floor-spaced manifest-ordered
members. This yields indices `[0, 11, 33, 51, 55, 74, 85, 87, 106]`.

A candidate is eligible only if:

1. its artifact/config hashes, schema, dtypes, and finite-value checks pass;
2. its text, audio, and total losses are finite on the complete fixed
   validation manifest;
3. the pinned official server and exact local client run all nine rows;
4. every row has finite decoded audio, RMS at least 0.001, nonempty text, and
   Persian-script fraction at least 0.5;
5. all nine rows pass; no cascade, Piper, Qwen, or formant fallback is used.

Among eligible candidates, select minimum complete-scope total validation loss;
an exact tie selects the earlier step. Gates cannot be weakened after outputs
are observed. If none is eligible, v3 fails closed and the next experiment must
change training/data design under a new predeclared protocol.

## Final-test firewall

The same 738-row / 14-session v2 final-test manifest may be reused because v3
uses exactly the same group-disjoint train and validation manifests and the
test has never been accessed. No final-test row, objective, generation, or
metric may be read before v3 selection is frozen.

If and only if one v3 candidate is eligible, validate the exact selected
adapter and access the final test once. Final outcomes may never revise
checkpoint selection. If no candidate is eligible, do not run any final-test
stage.

## Storage bound

The v2 rank-64 LoRA tensors occupy about 0.361 GiB in BF16; doubling the rank
estimates about 0.722 GiB per embedding-free v3 adapter. One probe plus five
candidates therefore requires about 4.33 GiB, excluding transient-save and
safety headroom. Preflight must measure current free space and require the
estimated six retained adapters, one transient adapter, and at least 2 GiB of
additional safety space before launch.
