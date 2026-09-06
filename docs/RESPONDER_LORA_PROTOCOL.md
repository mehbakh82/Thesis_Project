# Frozen compact-responder LoRA protocol

Protocol frozen before training `responder_lora_v1` and before measuring its
development or final semantic scores.

## Purpose and claim boundary

The existing NeMo -> Qwen2.5-0.5B -> Piper cascade passed its mechanical
nine-turn validation panel, but its subsequently frozen automatic semantic
proxy was strongly negative. This experiment asks whether supervised LoRA
adaptation of the same compact responder on the project's authorized Persian
dialogue pairs improves held-out response relevance while retaining a small
4090-compatible architecture.

This is an automatic engineering experiment. It cannot substitute for human
Persian ratings, an independent dialogue benchmark, factuality/safety testing,
or physical 4090/browser latency. The data labels are automatically aligned
natural podcast turns under the documented QA waiver; they are not manually
verified instruction-response annotations. Internal training is supervisor
approved, but source licensing and redistribution remain unverified. No source
text, audio, or trained adapter is committed or redistributed.

## Locked data and separation

- Source manifest:
  `data/processed/manifests/conversations.jsonl`, SHA-256
  `aabf12268cce2854ef8b16ffd9339c6d703339c6b4c50a23a105447047a13a86`.
- Exported train manifest: 6,419 rows, SHA-256
  `a84c9f1f924e8544499bd502413c469d30a10b98ceefc4e5a1f853217af0e517`.
- Exported validation manifest: 131 rows, SHA-256
  `a2762495830c82ce12d3dc4cc8469e2110ae21a77ea91f9b70498a6812e187a7`.
- Training starts from rows whose frozen split is `train`, then keeps the 2,576
  rows whose first nonempty response sentence is complete within 64 tokenizer
  tokens. This content-independent length rule avoids teaching an artificial
  end-of-turn token after a truncated sentence.
- Checkpoint selection uses only the first two validation sessions after sorting
  sessions by SHA-256 of `20260906-responder-lora-v1`, a NUL byte, and the
  session ID. Their hashes/counts are:
  `20b9c9f30f140385942bc1311733ef0fcaa7a892f83cbca5e04b308984886f27`
  (22 rows) and
  `0fe76c73675252a81c2f14ce4dd1ed6e5fc96d5903130f945996c9ee671631ef`
  (17 rows).
- Final semantic evaluation uses the other three validation sessions and the
  following 40 content-blind, hash-ranked row indices:
  `20, 21, 28, 29, 30, 31, 34, 36, 38, 39, 41, 42, 46, 49, 51, 52, 55, 58,
  59, 85, 86, 87, 89, 90, 91, 93, 94, 102, 104, 105, 106, 109, 111, 114,
  117, 119, 120, 123, 124, 127`.
- Those 40 indices exclude the already observed semantic-baseline indices
  `0, 16, 32, 48, 65, 81, 97, 113, 130`.
- The 204-row test split is not used for training, checkpoint selection, model
  development, or this final evaluation. Before protocol freeze, an aggregate
  schema/length/leakage audit parsed the combined manifest, including test-row
  metadata and text lengths; no test transcript or response was displayed or
  manually inspected. Therefore the narrower claim is "test not used," not
  "test bytes never read."
- Session IDs are group-disjoint across train, validation, and test. Exact user
  and user/response-pair hashes have zero cross-split overlap.

## Locked model and optimization

- Base: locally cached `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- System prompt: the cascade's first Persian-only short-response prompt.
- Assistant-only causal loss; system/user tokens and padding are masked.
- Each user is capped at its last 256 tokenizer tokens. Each reference response
  is reduced to its first nonempty sentence and capped at 64 tokenizer tokens,
  matching the deployed short-answer contract. Maximum sequence length is 512.
- LoRA rank 16, alpha 32, dropout 0.05, no bias, applied to `q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
- FP32, seed 20260906, micro-batch 8, gradient accumulation 4, effective batch
  32, AdamW, learning rate 0.0002, weight decay 0.01, gradient norm 1.0,
  5% linear warm-up followed by linear decay, exactly two epochs.
- Base loss and loss after each epoch are measured on the 12 development rows
  satisfying the identical complete-response rule; the 39-row session partition
  itself remains fixed before filtering.
  The selected checkpoint is the epoch with the lowest finite development
  assistant-token loss. No semantic score participates in selection.
- Only adapter/tokenizer files are written locally under
  `checkpoints/responder_lora_v1`; they are ignored by Git and are not uploaded.
  A committed privacy-safe report retains hashes, sizes, parameters, losses,
  environment identity, and the selected relative path.

The training run is valid only if all locked hashes/counts/separation gates
pass; every training and development row is authorized for internal training;
no test row enters either loader; the exact base revision is present; all
losses are finite; exactly two epochs complete; and an adapter tree is hashed.

The first frozen execution at commit `5a4f472d98599b29ad8a32055859e80d24dab126`
stopped before any checkpoint or semantic measurement: BF16 backward produced a
non-finite gradient at reproduced batch 48/optimizer step 11. The failure is
retained in `results/training/responder_lora_v1_attempt1_failed.json`. The only
correction is FP32 compute; data, split, optimizer hyperparameters, panel, judge,
and success thresholds remain frozen.

## Locked final semantic comparison

The evaluator runs the audited user channel 1 through the same NeMo ASR service.
It compares three blinded replies for every locked final-panel row:

1. the unchanged base Qwen responder;
2. the selected LoRA responder; and
3. the aligned next-speaker reference response (a data-quality comparator, not
   a gold or human-verified answer).

The exact local `Qwen/Qwen3.8-27B-FP8` revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, served as `qwen3.8-27b` from
image `sha256:ffb2d59b1c059a5bd8d781320c9f5189de8293693b7d95da54befddaa54abf52`,
scores relevance and coherence from 0 to 4. It is different weights but the
same model family as the responder, which is an explicit limitation. The judge
gets no arm name, model identity, expected result, row index, reference status,
or threshold. Arm order is deterministically permuted per row. Each arm is
scored twice at temperature 0, top-p 1, seed 20260906, JSON-only output, and
reasoning disabled. No rationale is requested or retained.

The result is mechanically valid only if all 40 rows, all 240 calls, exact
model/container/artifact hashes, ASR success, base/adapter generation without
fallback, Persian-script constraints, deterministic arm mapping, and test-not-
used gates pass. Score magnitude is not a validity gate.

The predeclared automatic engineering success gate requires all of:

- mechanically valid evidence;
- selected-LoRA mean relevance at least 2.0/4;
- selected-LoRA mean coherence at least 2.5/4;
- LoRA-minus-base mean relevance gain at least 0.50;
- LoRA-minus-base mean coherence gain non-negative;
- at least 60% of rows have LoRA median relevance greater than base median
  relevance; and
- at least 70% of rows have LoRA median relevance at least 2.

Reference-arm scores contextualize whether the automatically aligned targets
are semantically usable. They do not alter the gate or select rows.

The evaluator stores only hashes, ordinal scores, aggregate statistics,
backend/artifact identities, and limitations. It stores no plaintext input,
reply, reference response, judge rationale, audio, or host-specific path. All
model traffic stays on loopback.
