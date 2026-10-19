# Responder candidate comparison v2 — protocol

Status: frozen before any v2 reply was generated.

## Why a v2

v1 judged with a same-family judge (Qwen3.8 rating Qwen responders), used only
podcast next-turn inputs, and did not test the target use case (an assistant
for older Persian speakers). v2 adds an elderly-assistant request panel, an
elderly-assistant system prompt, a third rubric dimension, Gemma 4 E4B, and a
second, cross-family judge.

## Panels (`scripts/evaluate_responder_candidates_v2.py --stage plan`)

- **Elderly requests:** 60 colloquial requests across eight categories
  (health/medication, family, daily life, information, emotional support, home
  safety, religion/culture, technology help), hand-written by the closeout
  author before generation (`configs/responder_v2_elderly_queries.json`).
- **Natural turns:** 60 user turns (4–30 words, caption text) from the 24
  balanced-v2 sessions that appear in no earlier panel and in no Moshi training
  or test split (Digiato/Zoomit), at most three per session (60 rows from 20
  sessions). Frozen panel file SHA-256:
  `c97c1e1761e6706976e9ad2afa681e1a9203843cbe135f8e3380584977619993`.

## Arms

| Arm | Model | System prompt |
|---|---|---|
| `qwen3_4b_prompt_v2` (incumbent) | Qwen3-4B-Instruct-2507 | deployed prompt-v2 |
| `qwen3_4b_elderly_v1` | Qwen3-4B-Instruct-2507 | elderly-assistant v1 |
| `qwen35_4b_elderly_v1` | Qwen3.5-4B (thinking disabled) | elderly-assistant v1 |
| `gemma4_e4b_elderly_v1` | Gemma 4 E4B (QAT) | elderly-assistant v1 |

All arms decode greedily (96 new tokens) with Latin-letter tokens masked, as in
the deployed streaming path. The Mizan-1.5-4B local server and Gemma 4 12B are
excluded: the former is configured without chat logits (an apparent routing
model), the latter's `gemma4_unified` architecture is not loadable in the
pinned transformers version.

## Judges and rubric

Each reply is scored in random order, without arm names, by (a) the local
Qwen3.8-27B judge and (b) Gemini 3.8 Flash, on relevance and coherence (the v1
rubric, verbatim) and elderly appropriateness (0–4). Gemma is a Google model
and Gemini a Google judge, Qwen arms and the Qwen judge share a family; a
promotion must therefore hold under **both** judges.

## Promotion rule

A challenger replaces the incumbent only if, under every available judge, its
paired mean relevance gain is at least +0.25, its coherence and
appropriateness gains are non-negative, and it wins at least 55% of rows on
relevance. Inter-judge exact relevance agreement is reported. If the Gemini
judge is unavailable, no promotion is declared final.
