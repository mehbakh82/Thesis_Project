# star_tts_plus

Measured inference improvements for the lab Persian VITS model
(`star.text_to_speech`) **without retraining**. The lab repository is imported,
never modified. With default `Options()` the engine reproduces the lab Triton
service exactly; every improvement is a separate option so its effect can be
measured alone.

| Option | What it fixes | Effect on the audio |
|---|---|---|
| `numbers=True` | Thousands separators (`500,000` was read «پانصد»), clock times, dates, percentages, decimals (also the Persian decimal slash), blood pressure / ratios, simple fractions, ranges, phone numbers digit by digit, ordinals (the lab gives «سیم» for 30th), Latin units after numbers, and the formal «یک هزار» / «یکصد» → «هزار» / «صد» | Changes only texts with digits |
| `tokenizer_fix=True` | The fork's number step re-tokenizes with hazm's verb-part joining, which drops a sentence-final «خواهید/خواهد» («چه چیزی می‌خواهید» is spoken «چه چیزی می») | Changes only the affected sentences |
| `noise_scale`, `noise_scale_dp`, `length_scale` | VITS sampling noise and speaking rate | Tuned on the development texts: `0`, `0.5`, `1.1` |
| `split="punct"`, `pause_ms` | Sentence handling: split also at «؟ ! ؛ :», chunks of up to 25 words, a pause between chunks | Tested; not kept (lower CER, slightly lower DNSMOS) |
| `persian_punct=True` | «؟ ؛ ،» are missing from the fork's punctuation set, so espeak-ng ends a clause there and the fork glues the next word onto the previous one («خوردید؟ بله» → one word) | Tested; not kept (slightly worse, presumably because the model was trained on the glued form) |
| `fast_espeak=True` | espeak-ng runs in-process through libespeak-ng (input handled byte for byte as the CLI does) instead of one process per sentence, and the two extra subprocesses the fork starts for every sentence (`--voices`, `--version`) are cached: front-end 34 → 10 ms per sentence | None: identical phoneme strings |
| `resample="kaiser_best_poly"` | The service's `kaiser_best` resampling, computed with resampy's own filter table expanded into a polyphase filter | None: same samples (SNR > 110 dB), about 10× faster |
| `cuda=True` | Acoustic model on the GPU | Same model; GPU numerics |

The frozen configuration and its evaluation are in the thesis repository:
`docs/TTS_LAB_VITS_OPTIMIZATION_PROTOCOL.md` and
`docs/TTS_LAB_VITS_OPTIMIZATION_RESULT.md`.

## Use

```python
from pathlib import Path
from star_tts_plus.engine import Options, StarTTS

tts = StarTTS(Path("/path/to/star.text_to_speech"), Options(cuda=True, **FROZEN))
audio_16k = tts.tts("ساعت ۱۲:۴۵ با پزشک قرار دارید.")
```

`StarTTS` needs the lab repository's environment (its Coqui fork, hazm 0.9.3,
kenlm, the ezafe checkpoint under `kasre/saved_checkpoints/`, the heteronym
language model) and the `espeak-ng` shared library (`libespeak-ng.so.1`).

## Triton service

The lab's Triton backend (`triton_models/tts_model/1/model.py`) can route
inference through `StarTTS` with the request/response contract unchanged; that
patch contains the lab's own service code and is handed to the lab with this
package rather than published here. It also rejects a batch of more than one
text explicitly (the original synthesized only the first text of a batch) and
can run a warm-up sweep at start-up (`StarTTS.warmup()`, which only fills
cuDNN's per-shape caches: a new input length otherwise costs about 70 ms once on
an H100).

## Tests

```bash
STAR_TTS_REPO=/path/to/star.text_to_speech python -m pytest -q tests
```

The number and front-end tests need the lab repository; the resampler and
splitter tests need only numpy, scipy, librosa and resampy.

## What remains outside this package

Everything that needs new training data or retraining: the ezafe model's
occasional errors (e.g. an ezafe after «مثلاً»), the acoustic model's own
pronunciation errors (e.g. suffixes after a zero-width non-joiner), and the
punctuation and tokenization quirks the model was trained with (fixing the
Persian punctuation at inference made the development results slightly worse).
