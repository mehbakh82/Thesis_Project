# Corpus label verification — result

Protocol: `docs/CORPUS_LABEL_VERIFICATION_PROTOCOL.md` (committed before any
call). Script: `scripts/verify_corpus_labels.py`. Receipt:
`results/eval/corpus_label_verification.json`. Labeler: Gemini 3.8 Flash, two
independently worded boundary prompts plus one verbatim transcript per pair, at
temperature 0. These are **independent automatic labels**, not human
verification.

## Sample

300 balanced-v2 pairs (150 per stratum) drawn from the 1,353 pairs whose source
audio is still on local disk after the 2026-10-06 offload. Strata by the
pipeline's overlap label at the turn boundary (an overlap interval within 1 s
of the end of the user turn): 2,428 boundary-overlap and 4,123 boundary-clean
pairs in the full 6,551-pair manifest. The local frame is mostly Digiato
(956) and Zoomit (281), with Mehran Rowshan 107 and Tabaghe16 9; the sample has
208 / 67 / 24 / 1. Corpus-level rates are weighted by the full-manifest strata.

## Results (agreed labels; Wilson 95% intervals)

| Measure | Boundary-overlap | Boundary-clean | Weighted |
|---|---|---|---|
| Two-pass label agreement | 72.7% | 70.7% | — |
| New voice starts at the pipeline's turn boundary | **94.5%** (88.5–97.5) | **93.4%** (87.0–96.8) | **93.8%** |
| Gemini hears overlap at the boundary | **75.7%** (66.9–82.7) | 32.5% | — |
| Gemini hears no overlap at the boundary | — | **67.5%** (58.8–75.1) | — |

Agreed boundary labels, boundary-overlap stratum: interrupt 80, turn change 16,
backchannel 7, none 6; boundary-clean: turn change 60, interrupt 35, none 7,
backchannel 4.

Captions against Gemini transcripts (279 user turns of at most 60 s; 21 longer
turns excluded): corpus CER **15.9%**, median per-turn CER 16.5%, and 58.4% of
turns at or below 20%. Neither side is ground truth: part of the difference is
spelling variation in colloquial Persian, part is caption error.

## Reading

- **Turn boundaries are reliable:** at about 94% of pipeline boundaries an
  independent listener hears a different person start talking, in both strata.
- **Overlap labels are moderately reliable:** three quarters of the marked
  overlaps are heard as overlap, but Gemini also hears overlap at a third of
  the boundaries the pipeline marks clean, which suggests that diarization
  misses short overlaps (the corpus under-reports overlap rather than
  inventing it).
- **Captions are noisy:** a median turn differs from a careful verbatim
  transcript by about one character in six, which bounds what the corpus text
  can teach a model; this is consistent with the YouTube-caption origin.

## Claim boundary

Automatic labels checked by another automatic listener on a convenience frame
(pairs whose audio is still local, mostly two of the four channels); no human
annotation. The corpus itself is unchanged.
