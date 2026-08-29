# Moshi v2 checkpoint-selection and final-test protocol

Frozen before v2 training: 2026-08-27.

Post-training complete-prompt correction frozen 2026-08-29; see `docs/MOSHI_V2_PROTOCOL_CORRECTION.md`. The original panel below is preserved as the superseded protocol.

## Why a second run is required

The completed v1 run remains valid evidence for teacher-forced loss learning, but
it failed the first official-server autoregressive validation smoke. With one
frozen validation input, the base Moshika emitted non-silent audio and 20 text
messages; step 500 emitted non-silent English audio/text; steps 1,000, 2,000,
4,000, and the loss-selected step 8,000 emitted the same near-silent response
and no text. The v1 selected adapter is therefore not deployment-eligible.

The v1 run used 20-second chunks to coexist on the shared H100. Kyutai warns
that reducing duration can make the model become silent more quickly. Only
29.4% of v1 train chunks contained a response onset. Restoring the documented
100-second duration raises that fraction to 81.1%. A real one-step probe with
the otherwise exact v2 shape passed at 22.8 GB peak while other GPU processes
remained untouched.

## Immutable v2 data decision

Run:

```bash
.venv/bin/python scripts/prepare_moshi_v2_splits.py
```

The deterministic report `results/moshi_v2_split.json` freezes a new final
test before v2 training by holding out complete `source_session_id` groups
whose BLAKE2b-64 value is bucket 0 modulo 10. It contains 738 rows / 14
sessions / 11.894073 hours. The remaining v2 train set contains 5,681 rows /
138 sessions / 91.985640 hours. The unchanged validation set has 131 rows / 5
sessions / 3.431446 hours. All three session sets are pairwise disjoint.

The old v1 test split is not an input to v2 training, selection, or final
testing. It remains v1-only evidence.

## Candidate set

The sole scientific launch configuration is `configs/moshi_h100_v2.yaml`:
100-second duration, batch 1, four microbatches, rank-64 LoRA with embedding
tuning, maximum learning rate 2e-6, and 2,000 steps. Every 100-step adapter is
retained, yielding the predeclared candidates 100, 200, ..., 2,000.

A candidate is eligible only if all of these validation-only checks pass:

1. Its adapter and config hashes, shapes, dtypes, finite values, and official
   CUDA loader pass fail-closed validation.
2. Its text, audio, and total teacher-forced losses are finite on the same
   complete ordered v2 validation manifest.
3. On each frozen official-server validation-panel row
   `[0, 16, 32, 48, 64, 80, 96, 112, 130]`, transport and provenance pass,
   decoded PCM is finite, decoded RMS is at least 0.001, at least one text
   token is emitted, and at least 50% of Persian-or-Latin letters in decoded
   SentencePiece token text are Persian-script letters.
4. No cascade, Piper, Qwen, or formant fallback appears in the direct command.

Among eligible candidates, choose minimum complete-validation total loss;
exact ties choose the earlier step. If no candidate is eligible, v2 fails and
no adapter is promoted. Generated audio is not retained and no automatic gate
is called a perceptual, naturalness, relevance, or human-quality result.

## Untouched final evaluation

After selection is frozen, evaluate the selected hash exactly once on the new
v2 final test. The complete teacher-forced report compares the selected,
sign-flipped-LoRA, and base models on identical cached tokens. The script
refuses to overwrite an existing final-test report.

Report autoregressive automatic diagnostics separately on nine deterministic
rows selected without inspecting model output:

```text
floor(i * (row_count - 1) / (panel_size - 1)), i = 0..8
```

For the frozen 738-row test this is
`[0, 92, 184, 276, 368, 460, 552, 644, 737]`. These diagnostics use the same
speech/text/Persian gates but never select or revise a checkpoint. Human
listening and physical-4090 browser evidence remain separate gates and cannot
be replaced by this protocol.

This final-runtime reporting rule was frozen on 2026-08-29, after the clean
training retry launched but before checkpoint selection or final-test access.
It does not change the candidate set, validation eligibility gates, or selector.

## Post-training complete-prompt addendum

The training run completed, but the original panel was found to stream
unfinished five-second prefixes for eight of nine rows. That makes silence an
ambiguous and often correct turn-taking result rather than evidence that the
model cannot respond. Its generated reports and fail-closed selection are
preserved under explicit `truncated_prompt` names.

Before any corrected generation or final-test access, the replacement
validation rule was frozen in `docs/MOSHI_V2_PROTOCOL_CORRECTION.md`. It uses
only input PCM timing: retain rows whose entire user channel ends within five
seconds, then choose nine floor-spaced manifest-ordered members. The resulting
validation indices are `[0, 11, 33, 51, 55, 74, 85, 87, 106]`.

Candidate steps, loss criterion, tie break, official runtime, artifact gates,
RMS threshold, text gate, Persian-script threshold, and the 9/9 requirement
remain unchanged. The exact corrected indices were not predeclared before
training, and that limitation must be reported. The final runtime diagnostic
uses the same input-only rule after selection; the superseded fixed final-test
indices are not used.
