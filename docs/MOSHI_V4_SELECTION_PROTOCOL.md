# Moshi v4 text-embedding-only checkpoint protocol

Frozen: 2026-08-30, after v3 selection failed closed and before any v4 probe,
optimizer step, generation, or access to the frozen final-test rows.

## Evidence and hypothesis

V2 used rank-64 LoRA with `ft_embed: true`. That broad upstream switch trained
all 25 embedding tensors: `text_emb`, `depformer_text_emb`, 16 main audio
embeddings, and seven Depformer audio embeddings. V2 briefly reached 2/9
runtime rows at step 400, then increasingly produced no speech or text even as
teacher-forced loss continued improving.

V3 restored rank-128 LoRA with `ft_embed: false`. It completed steps 100–500 and
improved identical complete-scope validation loss from 2.084004 to 1.879258,
but runtime pass counts were `[0, 0, 0, 1, 0]` out of nine. Most surviving text
remained English; the only passing row was at step 400. V3 selection returned
null and `test_access_started=false`. Canonical v3 evidence is committed at
`76bcf959741eb5637fa8114b3892bf00d1483db1`.

V4 tests one controlled change from v3: retain rank-128 LoRA and enable gradients
for exactly:

- `text_emb.weight`
- `depformer_text_emb.weight`

All 23 audio-codebook embedding tensors remain frozen. The upstream
`lora.ft_embed` flag remains false; the project launcher activates the exact
scope only when `MOSHI_TEXT_EMBEDDINGS_ONLY=1`. The machine-readable policy is
`configs/moshi_v4_embedding_policy.json`. The probe and every checkpoint must
contain all LoRA tensors plus exactly these two embedding tensors and no other
embedding tensor.

This is a falsifiable attempt to retain v2's limited Persian adaptation without
its broad audio-embedding drift. It is not a claim that embedding drift alone
caused the failures. Data, train/validation groups, 100-second context, effective
batch of four microbatches, loss weights, optimizer, learning rate, schedule,
seed, rank, and runtime gates remain identical to v3.

## Bounded candidate set

V4 is limited to 500 steps because both v2 and v3 had their strongest runtime
behavior at step 400 and degraded afterward. The complete predeclared candidate
set is `[100, 200, 300, 400, 500]`; all five checkpoints must be retained. The
full configuration is `configs/moshi_h100_v4.yaml`.

Before training, `configs/moshi_h100_v4_probe.yaml` must complete exactly one
optimizer step and one CPU-offloaded checkpoint under the same selective policy.
The probe must prove the exact 676-tensor schema (674 LoRA plus two text
embeddings), finite loss, expected GPU memory, and absence of every audio
embedding from the adapter. It is feasibility evidence only and is never a
selection candidate.

## Selection gates

Selection uses only the unchanged v2/v3 validation manifest. Retain rows whose
complete user-channel PCM ends within the five-second stream, then select nine
floor-spaced manifest-ordered members. The frozen indices remain
`[0, 11, 33, 51, 55, 74, 85, 87, 106]`.

A candidate is eligible only if:

1. its hashes, saved config, exact selective schema, BF16 dtypes, and finite
   values pass;
2. text, audio, and total losses are finite on the identical complete 202-chunk
   validation scope;
3. the pinned official server and attested local client run all nine rows;
4. every row has finite decoded audio, RMS at least 0.001, nonempty text, and
   Persian-script fraction at least 0.5;
5. all nine rows pass with no cascade, Piper, Qwen, formant, or other fallback.

Among eligible candidates, select minimum complete-scope total validation loss;
an exact tie selects the earlier step. No threshold, panel member, candidate,
or selection rule may change after outputs are observed. If no candidate is
eligible, v4 fails closed.

## Final-test firewall

The untouched 738-row / 14-session final test remains group-disjoint from the
unchanged train and validation manifests. No final-test row, objective,
generation, or metric may be read before selection is frozen and its exact
artifacts are committed from a clean worktree.

If and only if v4 produces an eligible checkpoint, statically validate and load
the exact selected adapter, commit the selection evidence, then access the final
test once. Test outcomes can never revise selection. If selection is null, no
final-test stage may run.

## Storage and stopping rules

Rank-128 LoRA tensor data occupy 775,749,632 bytes. The two BF16 text embeddings
add 327,690,240 bytes, estimating 1,103,439,872 tensor bytes per v4 adapter.
Preflight must require one probe, five retained candidates, one transient save,
and at least 2 GiB additional free-space safety headroom.

Training must start from the pinned base model, not a v2/v3 checkpoint. Stop and
record failure on non-finite loss, schema mismatch, ENOSPC, OOM, missing journal
scope marker, missing checkpoint, or any final-test access. A clean 500-step run
must still be rejected if no candidate passes all runtime gates.
