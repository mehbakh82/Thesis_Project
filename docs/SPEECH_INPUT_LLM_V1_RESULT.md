# Speech-input LLM versus cascade (W3) — result

Protocol: `docs/SPEECH_INPUT_LLM_V1_PROTOCOL.md`. Receipt:
`results/eval/speech_input_llm_v1.json`. Judge: local Qwen3.8 (temperature 0),
replies shuffled per arm; 60 natural user turns + 60 simulated-elderly spoken
requests.

## Results

| Arm | Relevance natural | Relevance elderly | Relevance pooled | Coherence | Elderly appropriateness | First token after audio (median) |
|---|---|---|---|---|---|---|
| NeMo → Qwen3-4B prompt-v2 (production) | **2.85** | 2.75 | **2.80** | 3.36 | 2.62 | 86 ms |
| NeMo → Gemma 4 E4B text, elderly prompt | 2.03 | **2.95** | 2.49 | 3.69 | **3.15** | 98 ms |
| Gemma 4 E4B hears the audio (zero-shot) | 1.65 | 2.32 | 1.98 | 3.60 | 2.85 | **67 ms** |
| Gemma 4 E4B + speech LoRA | 1.97 | 2.13 | 2.05 | **3.81** | 2.85 | 200 ms* |

Paired bootstrap 95% intervals of relevance minus production: zero-shot
[−1.13, −0.50]; LoRA [−1.05, −0.46] pooled, [−1.27, −0.45] natural,
[−1.05, −0.18] elderly.

\* The adapter was served unmerged through vLLM's LoRA path; the extra
≈130 ms is serving overhead, not a property of the model.

The LoRA (rank 16, 34.9 M parameters; 3,817 user turns = 11.2 h from 137
training sessions; transcription + self-distillation from Gemma's own
text-mode replies) lowered the held-out loss from 1.296 to 0.712. The
lowest-loss checkpoint was the last one (step 455).

## Decision (rule fixed before the LoRA was trained)

**Negative.** The best speech-input arm (LoRA, pooled relevance 2.05) is 0.75
below the production cascade, far outside the 0.10 margin, so the direct
speech-input route is neither a positive result nor adopted. The cascade stays.

## What the comparison shows

- **Skipping ASR costs relevance even with the same LLM.** Gemma answering the
  NeMo transcript scores 2.49; Gemma hearing the audio scores 1.98. The
  bottleneck is Gemma's Persian speech understanding (it confuses words such as
  «قند» and «فشار خون»), not its Persian generation.
- **Self-distillation helps in-domain and hurts out of domain.** On real
  conversational speech the LoRA recovered most of the gap to its own
  text-mode teacher (1.65 → 1.97, teacher 2.03). On the synthetic slow-voice
  elderly requests it got worse (2.32 → 2.13): 11 h of podcast speech adapts
  the model to that domain rather than to Persian speech in general.
- **The latency gain is small.** The NeMo CTC ASR adds only about 20 ms, so
  removing it saves little (67 vs 86 ms to first token) while losing a lot of
  quality.
- Gemma's text mode with the elderly prompt is better suited to the elderly
  requests (relevance 2.95 vs 2.75, appropriateness 3.15 vs 2.62) but much
  worse on natural conversational turns; responder v2 found the same pattern.

## Claim boundary

One automatic LLM judge, 120 requests, simulated elderly voices, H100 timing.
The natural-turn references are YouTube captions. No human ratings.
