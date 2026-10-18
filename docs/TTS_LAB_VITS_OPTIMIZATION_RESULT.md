# Lab VITS inference optimization — result

Protocol: `docs/TTS_LAB_VITS_OPTIMIZATION_PROTOCOL.md` (amendments 1 and 2,
each committed before the steps it adds were run). Code:
`tools/star_tts_plus/` (the inference package; the lab repository is imported,
never modified), `scripts/tune_tts_lab_vits.py` (development steps),
`scripts/synthesize_tts_star_vits.py`, `scripts/assemble_tts_candidates_v3.py`
and `scripts/score_tts_candidates_v2.py` (evaluation). Receipts under
`results/eval/`: `tts_lab_vits_dev_serving.json` (first check, failed),
`tts_lab_vits_dev_serving_v2.json`, `tts_lab_vits_dev_{grid,split,punct,tokenize}.json`,
`tts_lab_vits_optimized_eval.json`.

**Starting point** (TTS v3 supplement, the lab service as it is,
`tts_candidates_v3_with_lab_vits_pinned.json`): two-listener CER 9.89% against
Mana's 5.82% (paired 95% interval of the per-clip difference [+1.4, +8.1] pp:
significantly less intelligible), DNSMOS OVRL 3.39 against 3.13, first four
words in 538 ms on CPU and 187 ms with the acoustic model on the H100.

## Evaluation (60 TTS v2/v3 texts, synthesized once after freezing)

Frozen configuration (`frozen_options.json` of the run): numbers on, tokenizer
fix on, in-process espeak-ng, exact `kaiser_best` resampling, acoustic model on
the GPU, `noise_scale` 0, `noise_scale_dp` 0.5, `length_scale` 1.1, the
service's sentence split, Persian punctuation as the fork handles it.

| Voice | Two-listener CER | Δ vs Mana, 95% CI | NeMo / Seamless CER | DNSMOS OVRL | First four words (median) |
|---|---|---|---|---|---|
| Mana (deployed) | **5.82%** | — | 6.34% / 5.31% | 3.13 | 84 ms CPU |
| ganji_adabi | 6.93% | [-1.1, +4.9] pp | 5.59% / 8.27% | 3.40 | 92 ms CPU |
| lab VITS, as served | 9.89% | [+1.4, +8.1] pp | 8.36% / 11.42% | 3.39 | 538 ms CPU, 187 ms H100 |
| **lab VITS, optimized** | 9.19% | [+0.8, +8.1] pp | 7.50% / 10.88% | **3.43** | 416 ms CPU, 116 ms H100 |

The optimization improves the lab voice on every measure — CER
9.89% → 9.19% (both listeners lower), the highest DNSMOS of all
voices, and first audio 538 → 416 ms on CPU and 187 → 116 ms on the H100. The
absolute CER gain matches the development gain (−0.7 pp against −0.8 pp), but
the gap to Mana was +4.1 pp, so the voice remains **significantly less
intelligible than Mana** (the interval excludes zero). Most of the remaining gap
is the Seamless listener (10.9% against 5.3%).

## Decision (TTS v3 rules, unchanged)

- **The optimized lab VITS does not enter the listening study**: it is
  significantly less intelligible than Mana, although its DNSMOS is higher. The
  study keeps Mana, ganji_adabi and the human anchor.
- It cannot replace Mana. Its median first-four-words time on the H100
  (116 ms) is also above the 100 ms deployment limit.
- What remains between this voice and Mana is the acoustic model and its
  training front-end (see the dev error analysis below), not inference.

**Post-hoc serving note (development texts, not part of the decision).** The
H100 time of a *new* input length includes a one-time cuDNN set-up of about
70 ms (a repeated four-word text takes 45 ms). A start-up warm-up sweep
(`StarTTS.warmup()`, 340 short phrases, 35 s) leaves every output bit-identical
and brings the median for fresh first-four-words texts to 50 ms (p90 122 ms).

**Remaining development errors** (word-level differences at the selected
setting) are pronunciation details: ezafe vowels that the listeners write as a
different word («نام» → «نامه», «متن» → «متنی»), suffixes after a zero-width
non-joiner («جمله‌تان»), and weak syllables. Fixing them needs a pronunciation
lexicon or retraining.

## Development steps

Development texts: 339 replies of the other responder-v2 arms (none is an
evaluation text); steps 3, 4, 4b and 2b use the 313 without digits. Every text
has a fixed seed, identical across configurations. Selection metric: the
two-listener CER (NeMo v4 CTC, Seamless v5ft; `fa-asr-score-v1`).

### Step 1 — serving (must not change the audio)

| Change | Criterion (development texts) | Result |
|---|---|---|
| espeak-ng in-process (CLI-exact input) and cached `--voices` / `--version` queries | identical phoneme strings on every sentence chunk | 909/909 (fork tokenizer), 909/909 (with the step 2b fix); front-end 34.3 → 9.9 ms per chunk |
| `kaiser_best` computed exactly as a polyphase filter (resampy's own table) | SNR against `kaiser_best` ≥ 40 dB on every clip | minimum 113 dB (float rounding), 158 → 16 ms per clip |
| soxr (the protocol's first candidate) | same | failed: minimum 37.6 dB; not used |
| acoustic model on the GPU | — | adopted |

The fork also builds a new `ESpeak` object for every sentence, and each
construction ran `espeak-ng --voices` and `espeak-ng --version` (about 14 ms);
these answers are now cached.

### Step 2 — number and time reading

The fork reads digits with `num2fawords` per token. Wrong or unnatural
readings, fixed in `star_tts_plus/numbers.py` and checked by 30 unit cases:

- thousands separators: `500,000 تومان` was spoken «پانصد تومان» (every group
  after the separator was dropped) → «پانصد هزار تومان»;
- clock times `12:45` → «دوازده و چهل و پنج دقیقه», `2:30` → «دو و نیم»;
- dates `1403/05/12` → «دوازدهم مرداد هزار و چهارصد و سه» (Solar Hijri or
  Gregorian month names by year) instead of «… اسلش …»;
- blood pressure and ratios `120/80` → «صد و بیست روی هشتاد»; simple fractions
  `1/2` → «یک دوم»; the Persian decimal slash `37/5` → «سی و هفت و پنج دهم»;
- decimals, percentages, ranges (`10-15` → «ده تا پانزده»), phone numbers digit
  by digit, ordinals (the fork gives «سیم» for 30th → «سی‌اُم»), Latin units
  after a number (`5mg` was spelled out letter by letter → «میلی‌گرم»);
- the formal «یک هزار …» / «یکصد …» → the spoken «هزار …» / «صد …».

In the development and evaluation texts these rules change only three clock
times; they matter for prices, dates, dosages and phone numbers in general use.

### Step 2b — tokenizer fix (amendment 2)

The fork's number step re-tokenizes every sentence with hazm's
`word_tokenize`, whose verb-part joining treats «خواهید/خواهد» as the future
auxiliary: at the end of a sentence the word is **dropped** («چه چیزی
می‌خواهید» is spoken «چه چیزی می»). Tokenizing without verb joining changes 5 of
the 909 development sentence chunks.

| On top of the step 4b result | NeMo CER | Seamless CER | Two-listener CER | DNSMOS OVRL |
|---|---|---|---|---|
| fork tokenizer | 3.68% | 2.68% | 3.18% | 3.4221 |
| **without verb joining** | **3.54%** | **2.58%** | **3.06%** | 3.4218 |

Kept (rule: CER not worse). DNSMOS is 0.0003 lower; under the DNSMOS condition
of steps 4 and 4b the fix would have been rejected. The rule for this step was
committed before it was run.

### Step 3 — inference settings

Rule: lowest two-listener CER; among settings within 0.5 pp of it, the highest
DNSMOS OVRL. · marks settings outside the 0.5 pp band.

| noise_scale | noise_scale_dp | length_scale | NeMo CER | Seamless CER | Two-listener CER | DNSMOS OVRL |
|---|---|---|---|---|---|---|
| 0 | 0 | 1 | 3.59% | 2.68% | 3.13% | 3.415 |
| 0.333 | 0.5 | 1.1 | 3.75% | 2.61% | 3.18% | 3.415 |
| 0 | 0 | 1.1 | 3.59% | 2.79% | 3.19% | 3.418 |
| **0** | **0.5** | **1.1** | 3.71% | 2.70% | **3.21%** | **3.422** |
| 0.333 | 0 | 1 | 3.81% | 2.70% | 3.25% | 3.406 |
| 0.333 | 0 | 1.1 | 3.78% | 2.73% | 3.26% | 3.409 |
| 0 | 0.5 | 1 | 3.72% | 2.81% | 3.27% | 3.421 |
| 0.333 | 0.5 | 1 | 3.81% | 2.81% | 3.31% | 3.415 |
| 0.667 | 0 | 1 | 3.96% | 3.00% | 3.48% | 3.385 |
| 0.667 | 0.5 | 1.1 | 4.12% | 2.90% | 3.51% | 3.393 |
| 0 | 1 | 1 | 4.13% | 3.18% | 3.66% · | 3.420 |
| 0.667 | 0.5 | 1 | 4.14% | 3.21% | 3.68% · | 3.392 |
| 0.667 | 0 | 1.1 | 4.13% | 3.23% | 3.68% · | 3.388 |
| 0.333 | 1 | 1 | 4.26% | 3.12% | 3.69% · | 3.409 |
| 0 | 1 | 1.1 | 4.25% | 3.15% | 3.70% · | 3.421 |
| 0.333 | 1 | 1.1 | 4.32% | 3.21% | 3.77% · | 3.409 |
| 0.667 | 1 | 1 (model default) | 4.47% | 3.27% | 3.87% · | 3.390 |
| 0.667 | 1 | 1.1 | 4.75% | 3.55% | 4.15% · | 3.386 |

Selected: `noise_scale` 0, `noise_scale_dp` 0.5, `length_scale` 1.1 (the highest DNSMOS among the 10 settings within 0.5 pp of the best CER, 3.13%).

The model's sampling noise costs intelligibility: removing the latent noise
(`noise_scale` 0.667 → 0) and halving the duration noise lowers the CER from
3.87% to 3.21%; the selected setting also speaks 10% slower.

### Step 4 — sentence handling

| Sentence handling | Two-listener CER | DNSMOS OVRL |
|---|---|---|
| **service split** (at «.» «،», 10-word chunks, no pause) | 3.21% | **3.422** |
| punctuation-aware (also «؟ ! ؛ :», ≤25 words, 120 ms pause) | **3.13%** | 3.412 |

Rule: kept if the CER is not worse **and** DNSMOS is not lower. The CER is
lower but DNSMOS is lower too, so the service's split stays.

### Step 4b — Persian punctuation (amendment 1)

«؟ ؛ ،» are not in the fork's punctuation set, so espeak-ng ends a clause
there and the fork joins the clause outputs without a space («خوردید؟ بله» is
phonemized as one word; 51 of 850 development chunks contain such a mark
inside the chunk). Mapping them to «? ; ,» before the front-end:

| | Two-listener CER | DNSMOS OVRL |
|---|---|---|
| **as the fork does** | **3.21%** | **3.422** |
| «؟ ؛ ،» → «? ; ,» | 3.27% | 3.419 |

Not kept: both are slightly worse, presumably because the model was trained
through the same front-end and learned from the joined form.

## Deviations and disclosures

- **espeak-ng in-process instead of a persistent process** (same intent, same
  output). The first in-process version kept the newline that the CLI strips,
  so 5 of 909 development chunks differed in one stress mark and the first
  serving check failed (receipt kept); amendment 2 made the input CLI-exact.
  Steps 3, 4 and 4b ran with the first version: the 5 chunks differed alike in
  every configuration, so the comparisons inside those steps are unaffected,
  but their absolute CERs differ slightly from the fork's front-end (the
  selected setting: 3.21% then, 3.18% with the CLI-exact input).
- **Resampler** (amendment 1): soxr, named in the protocol, failed the 40 dB
  criterion (minimum 37.6 dB, the difference lying above 7 kHz; this first check
  used audio derived from the TTS v3 evaluation clips — a signal-equivalence
  check, no quality measure was computed). A polyphase approximation of the
  Kaiser design then failed on the development texts (minimum 24 dB), and the
  exact expansion of resampy's own filter table passed.
- **Package versions.** The step 3 grid was synthesized before three package
  changes: the extended number rules and the cached espeak-ng queries (neither
  affects texts without digits or the audio) and the resampler's output length
  (only the last sample before trimming). The evaluation ran with the final
  package.
- **Device.** The optimized configuration is synthesized with the acoustic
  model on the GPU (part of the serving change); the as-is arm of TTS v3 was
  synthesized on CPU, as the service runs. Same model, same front-end
  dependencies (hazm 0.9.3).
- **Step 2b rule.** A correctness fix judged on intelligibility only (see
  above); stated in amendment 2 before the run.

## Claim boundary

Round-trip ASR intelligibility and DNSMOS are automatic proxies; the acoustic
model is as trained; naturalness is decided by the listening study.
