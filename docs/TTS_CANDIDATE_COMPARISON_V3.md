# TTS candidate comparison v3 — result

Protocol: `docs/TTS_CANDIDATE_COMPARISON_V3_PROTOCOL.md` (written before any
v3 audio). Receipt: `results/eval/tts_candidates_v3.json`. Synthesis:
`scripts/synthesize_tts_xtts.py`, `scripts/assemble_tts_candidates_v3.py`;
scoring: `scripts/score_tts_candidates_v2.py --experiment tts_candidates_v3`.

## Results (60 replies, two ASR listeners)

| Voice | Two-listener CER | Δ vs Mana, 95% CI | NeMo / Seamless CER | DNSMOS OVRL | First four words | Device |
|---|---|---|---|---|---|---|
| Mana (deployed) | **5.82%** | — | 6.34% / 5.31% | 3.13 | 84 ms | CPU |
| ganji_adabi | 6.93% | [−1.1, +4.9] pp | 5.59% / 8.27% | **3.40** | 92 ms | CPU |
| ParsVoice-XTTS (ManaTTS voice, zero-shot) | 13.55% | [+0.9, +19.3] pp | 8.04% / 19.06% | 3.14 | 1,751 ms (streaming first chunk 307 ms) | H100 |

The lab VITS model (`star.text_to_speech`) could not be run: its Persian
front-end needs an ezafe-recognition checkpoint that is not in its repository,
its DVC remote, or on this machine. Its acoustic weights and heteronym model were
retrieved and verified.

## Decision (pre-registered rules)

- **ParsVoice-XTTS does not enter the listening study**: it is significantly less
  intelligible than Mana (interval above zero). It could not be deployed anyway:
  even streaming, its first audio chunk takes 307 ms on an H100 (Piper: under
  100 ms on CPU), which alone would push the measured end-to-end latency past
  500 ms.
- **ganji_adabi stays in the study** (tied on intelligibility, higher DNSMOS);
  the study compares Mana and ganji_adabi with the human anchor.

## Why XTTS loses on intelligibility

On a typical clip XTTS is as intelligible as Mana (median clip CER 5.6% for
NeMo, 6.2% for Seamless). But 5–6 of the 60 clips derail: one 37-character
reply became 7.3 s of audio with a babbled tail, and on another clip Seamless
transcribed 774 characters for a 120-character text — the autoregressive
instability known for XTTS. A deployed assistant cannot tolerate roughly one
derailed reply in ten, whatever the naturalness of the rest.

## Claim boundary

Automatic proxies; one reference voice; fixed seed. ParsVoice-XTTS was evaluated
for research only (Coqui Public Model License, non-commercial).
