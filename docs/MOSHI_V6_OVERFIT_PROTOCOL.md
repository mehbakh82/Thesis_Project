# Moshi v6 train-only Persian capacity diagnostic

Status: frozen before the first v6 optimizer step.

## Question

Can the pinned English Moshika base produce audible Persian autoregressively
after a bounded text-language bootstrap when its untied full text output
projection is trainable, or is the current lightweight adaptation path unable
to acquire Persian even in-sample?

This is an optimization-capacity diagnostic, not model validation. It uses no
validation or final-test row, and a positive result cannot be reported as
generalization or deployment readiness.

## Frozen inputs and intervention

- Exactly 32 rows are selected only from the frozen v2 training manifest by
  `scripts/prepare_moshi_v6_overfit.py`.
- The selection is mechanical: authorized, background-clean, no marked overlap
  or interruption, short complete user/assistant turns, short export, and at
  least 95% Persian letters among alphabetic target characters.
- The rows remain automatically transcribed and are **not human-clean**.
- The model starts from the unchanged pinned Moshika/Mimi/tokenizer weights.
- The documented one-GPU `MOSHI_DISTRIBUTED_BACKEND=gloo` compatibility path
  is mandatory because this host's pinned NCCL 2.21.5 crashes before model load.
- Rank-64 LoRA remains enabled. This was frozen before any optimizer step after
  an exact rank-128 probe reached the first AdamW-state allocation but could not
  fit beside a newly occupied 10.6 GiB GPU allocation. No data, output gate, or
  success threshold changed. Rank 64 still provides nearly 194 million LoRA
  parameters, plus the full text parameters, for this 32-row memorization
  diagnostic.
  Audio embeddings remain frozen.
- `text_emb.weight`, `depformer_text_emb.weight`, and the independent full
  `text_linear.frozen_W.weight` are trainable.
- The complete audio objective is multiplied by 0.1 for this language-bootstrap
  diagnostic; the text objective is not multiplied down.
- Duration is 12 seconds, effective batch is four, maximum is 200 steps, and
  candidates are steps 50, 100, 150, and 200.

The exact configuration and parameter policy are
`configs/moshi_h100_v6_overfit.yaml` and
`configs/moshi_v6_overfit_policy.json`.

## Positive criterion

A candidate is diagnostic-positive only if all integrity checks pass and the
official Moshi server is exercised on a deterministic nine-row panel from the
same 32 training rows. All nine rows must:

1. stream the complete user turn;
2. return finite decoded audio with RMS at least `1e-3`;
3. return at least four normalized text characters, with Persian-letter fraction
   at least `0.80`;
4. contain no Unicode replacement character; and
5. have normalized character error rate at most `0.75` against the equal-length
   prefix of that row's training target.

The fixed-scope teacher-forced total and text losses must also be finite, and
the selected candidate's text loss must be at least 30% below step 50 unless
step 50 itself already passes all nine autoregressive rows. The earliest fully
passing candidate is selected for the diagnostic conclusion.

## Decision rule after the diagnostic

- Positive: the tokenizer/model can acquire Persian under the expanded text
  scope. Freeze a separate full-corpus v6 curriculum and restart from the base;
  this diagnostic adapter is not promoted as the final model.
- Negative: stop spending H100 time on Moshika LoRA. Pivot the direct-model
  research path to a Persian-capable base or formally retain the verified
  NeMo/Qwen/Piper cascade as the deployable system.

No threshold may be relaxed after inspecting outputs. The v2-v5 held-out final
test remains sealed throughout this diagnostic.
