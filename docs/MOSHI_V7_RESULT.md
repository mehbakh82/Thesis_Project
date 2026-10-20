# Moshi v7a result: romanized, word-aligned inner monologue

Protocol: `docs/MOSHI_V7_ROMANIZED_ALIGNMENT_PROTOCOL.md` (committed before the
first optimizer step). Evidence: `results/moshi_v7a_overfit_data.json`,
`results/moshi_v7a_overfit_reevaluation.json`,
`results/moshi_v7a_overfit_runtime.json`, `results/moshi_tokenizer_audit.json`.

## Outcome

**Negative under the frozen all-nine rule**, with the largest autoregressive
improvement observed in any Moshi experiment.

| Experiment (same 32 rows, same nine-row panel) | Step 50 | Step 100 | Step 150 | Step 200 |
|---|---:|---:|---:|---:|
| v6 (Persian script, whole-turn span, sampled text) | 0/9 | 1/9 | 1/9 | 1/9 |
| v6.1 (v6 weights, greedy text) — controlled baseline | 0/9 | 0/9 | 1/9 | 3/9 |
| v6.2 (Persian script + text-input dropout) | 0/9 | 0/9 | 0/9 | 0/9 |
| **v7a (romanized phonemes + exact word timing, greedy text)** | **0/9** | **0/9** | **6/9** | **4/9** |

Teacher-forced in-sample text loss fell from 1.124 (step 50) to 0.0149 (step
150) and 0.0117 (step 200). From step 150 the official server returned audible
speech (decoded RMS ≈ 0.07–0.19) and fluent romanized Persian-like text on 8
of 9 rows. The three step-150 failures were prefix CER 0.76 and 0.91 (just
above the 0.75 limit) and one row that stayed silent at every checkpoint.

## Interpretation

- The measured tokenizer/frame-rate bottleneck is real: removing it, with
  every other v6 factor fixed, doubled the best in-sample pass count relative
  to the controlled v6.1 baseline and turned near-silent output into audible
  speech with text on almost every row.
- It is not the only obstacle: generated text still drifts from the
  memorized target despite near-zero teacher-forced loss, so a
  teacher-forcing/free-running gap (exposure bias) remains.
- This is an in-sample memorization diagnostic. Even a pass would not have
  shown generalization or deployability.
- External precedent: J-Moshi (Ohashi et al., Interspeech 2025,
  arXiv:2506.02979), the published adaptation of Moshi to Japanese, also found
  Moshi's English tokenizer inefficient for its language, replaced it with a
  32,000-word Japanese SentencePiece model and re-initialized the text
  embeddings, and then pre-trained on about 60,000 hours of Japanese spoken
  dialogue (J-CHAT; 128 V100 GPUs for 36 hours) before fine-tuning on 344 hours
  of stereo dialogue and 602 hours of synthetic dialogue. That is more than 500
  times this project's 108.6 hours; no public Persian conversational corpus of
  that scale exists. The tokenizer fix is therefore necessary but, as v7a
  shows, not sufficient at this data scale.

## Decision

As frozen in the protocol, a negative v7a stops Moshika training: v7b was not
started and the sealed v2–v5 test split remains unopened. The direct-model
claim stays negative for Moshika. The direct-speech research effort continues
with a speech-input LLM (Gemma 4 audio) under its own protocol.

The evaluator retained generated assistant speech under
`checkpoints/moshi_v7a_overfit/runtime_audio/`, but its file names were keyed
by panel row only, so each candidate overwrote the previous one: the retained
nine clips are those of the last evaluated checkpoint (step 200). The
predeclared secondary Persian ASR intelligibility measurement is therefore
reported for step 200 only, once the project ASR is re-selected. The frozen
evaluator was not modified after the run.

**Secondary measure (step 200, reported, not gated):** with the re-selected
ASR (NeMo v4 CTC), the generated speech has a mean equal-length-prefix CER of
0.88 against the Persian targets; 0 of 9 rows are at or below 0.75 and one
transcript is empty (`results/moshi_v7a_step200_intelligibility.json`,
`scripts/score_moshi_v7_intelligibility.py`). The audible, text-bearing output
is therefore not yet intelligible Persian speech.

Note (2026-10-17): commit messages were later edited (co-author trailers removed, content and dates unchanged). The protocol commit recorded in the runtime receipt, `a5d7b01`, is now `43df266`; the full mapping is in `results/release/commit_message_rewrite_20261017.json`.
