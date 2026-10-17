# Speech-input LLM versus cascade (W3) — protocol

Status: written while the zero-shot arms were still running and before any of
their outputs or scores were read. The LoRA arm and its decision rule are fixed
here before the adapter is trained.

## Question

Can a single speech-input language model (Gemma 4 E4B, which accepts audio
directly) replace the ASR → text-LLM half of the cascade for Persian spoken
requests, at equal or better reply quality and lower time to first token?
This is the proposal's "direct" route on a model family with native audio
input, as an alternative to Moshi (see `docs/MOSHI_V7_RESULT.md`).

## Panels (same row ids as responder v2)

- `natural_turns`: 60 real user turns cut from unseen sessions of the corpus
  (median 5.5 s, longest 34 s; the input is truncated to Gemma's 30 s audio
  limit by the server).
- `elderly_requests`: the 60 hand-written elderly-assistant requests of
  `configs/responder_v2_elderly_queries.json`, synthesised as simulated elderly
  speech: two male Piper voices (`amir`, `reza_ibrahim`) not used for the
  assistant, 0.85× tempo with pitch kept, and room noise at −25 dB. This is a
  simulation, not recordings of older speakers.

The reference for judging is the text of the request (caption text for the
natural turns, the written request for the elderly panel), so the judge sees
what the user meant, never what the ASR heard.

## Arms

| Arm | Path | Prompt |
|---|---|---|
| `cascade_nemo_qwen3_promptv2` | NeMo v4 ASR → Qwen3-4B (production) | prompt-v2 |
| `cascade_nemo_gemma_elderly` | NeMo v4 ASR → Gemma 4 E4B text | elderly v1 |
| `speechin_gemma_elderly` | audio → Gemma 4 E4B (no ASR) | elderly v1 |
| `speechin_gemma_lora` | audio → Gemma 4 E4B + speech LoRA | elderly v1 |

All arms decode greedily with the same Persian-only regex constraint and 96 new
tokens. The LoRA arm is trained only on user turns of the corrected Moshi v2
*training* split (validation/test sessions excluded; the natural-turn panel
comes from held-out sessions), on two tasks: transcription (caption text) and
DiVA-style self-distillation, whose target is Gemma's own text-mode reply (elderly
prompt) to the caption transcript of the same turn. No elderly-panel text or
voice is used in training.

LoRA settings (fixed before training): the QAT w4a16 checkpoint is dequantized
to bf16 for training, rank 16 (alpha 32, dropout 0.05) on the language model's
attention and MLP projections, audio encoder and embeddings frozen, AdamW
lr 1e-4 with 30 warm-up steps and linear decay, effective batch 16, one epoch,
loss on the assistant tokens only. Checkpoints are saved every 100 steps and the
one with the lowest loss on a 5% session-held-out slice of the training
manifest is evaluated — once, on both panels. The adapter is served by vLLM on
the same w4a16 base as the zero-shot arm.

## Measures

- Relevance, coherence, elderly appropriateness (0–4; erratum: originally
  written "1–5", but the responder-v2 rubric used throughout is 0–4) from the local Qwen3.8
  judge with the responder-v2 rubric, replies shuffled per arm; a Gemini judge
  is added when budget allows and is reported separately.
- Time to first token after the end of the user audio:
  ASR time + LLM time to first token (cascade) or LLM time to first token
  (speech input). Median and p90 per panel, loopback HTTP on the H100.

## Decision rule (fixed now)

The speech-input route (`speechin_gemma_lora`, or the zero-shot arm if it is
better) is reported as a **positive direct-route result** if, on the pooled 120
rows, its mean relevance is not lower than the production cascade's by more than
0.10 and its median time to first token is lower. It is **adopted as the
deployed responder** only if, in addition, its relevance is not lower on either
panel separately by more than 0.10. Otherwise the cascade stays and the result
is reported as negative or mixed, with the numbers.

## Claim boundary

Automatic judge scores on 120 requests (one LLM judge; family-different from
Gemma and Qwen3-4B only in size, a disclosed limitation), simulated elderly
voices, H100 timing. No human ratings.

## Addendum (2026-10-17, after the v1 decision; exploratory)

- **Second judge.** The four v1 arms' replies are also judged by Gemini 3.8
  Flash (as planned above, once budget allowed); reported separately, the v1
  decision stays with the original judge.
- **Gemma 4 12B.** It was skipped in v1 because the pinned transformers could
  not load its `gemma4_unified` architecture; vLLM 0.30 can. Two exploratory
  arms on the same panels and settings: `speechin_gemma12b_elderly` (12B hears
  the audio, zero-shot) and `cascade_nemo_gemma12b_elderly` (NeMo transcript →
  12B text). Both judges. Being added after the v1 result, they cannot change
  the v1 decision; they show whether a larger model of the same family closes
  the speech-understanding gap.
