# Responder candidate comparison v2 — result (two judges)

Protocol: `docs/RESPONDER_CANDIDATE_COMPARISON_V2_PROTOCOL.md`. Receipts:
`results/eval/responder_candidates_v2_two_judges.json` (both judges) and
`results/eval/responder_candidates_v2_qwen38.json` (first judge only, recorded
when the Gemini budget had run out). 120 requests (60 elderly-assistant
requests, 60 natural user turns) × 4 arms, each reply judged blind on a 0–4
scale by the local Qwen3.8-27B judge and by Gemini 3.8 Flash.

## Results

| Arm | Qwen3.8 relevance (gain vs deployed) | Gemini relevance (gain, 95% CI, row win rate) | Gemini elderly appropriateness |
|---|---|---|---|
| Qwen3-4B, prompt-v2 (deployed) | 3.29 | 2.38 | 2.02 |
| Qwen3-4B, elderly prompt | 2.91 (−0.38) | 2.83 (+0.44, 0.22–0.67, 47%) | 2.57 |
| Qwen3.5-4B, elderly prompt | 3.31 (+0.02) | 3.36 (+0.98, 0.71–1.23, 58%) | 3.55 |
| Gemma 4 E4B, elderly prompt | 3.00 (−0.29) | 3.54 (+1.16, 0.93–1.39, 60%) | 3.77 |

Exact agreement between the judges on relevance: 44.6%.

## Decision (pre-registered rule)

**No promotion.** A challenger had to gain at least +0.25 relevance, lose
nothing on coherence and appropriateness, and win at least 55% of rows under
**every** judge. Under Gemini, Qwen3.5-4B and Gemma 4 with the elderly prompt
both clear that bar by a wide margin; under Qwen3.8, neither does. The deployed
responder stays.

## Interpretation

- The two judges disagree about the deployed responder, not only about the
  challengers: Gemini rates its replies 2.38, the Qwen judge 3.29. Gemini
  prefers the warmer, safer elderly-prompt replies on both panels.
- Family bias alone does not explain the split: Qwen3.5-4B is from the same
  family as the deployed model and the Qwen judge, yet Gemini rates it almost
  a point higher while the Qwen judge rates the two equally.
- The earlier positive final semantic test of the deployed responder
  (relevance 3.150 vs 1.425 for the 0.5B baseline) was judged by the Qwen judge
  only; this comparison shows that its absolute level depends on the judge.
  Only human ratings could settle which judge is closer to Persian listeners.
- Qwen3.5-4B with the elderly prompt is the strongest replacement candidate: it
  never loses under either judge and gains strongly under the cross-family one.
