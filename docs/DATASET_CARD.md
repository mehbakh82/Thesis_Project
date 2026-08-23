# Dataset card (Persian speech corpus and planned duplex extension)

## Identity

- Name: `fa-s2s-duplex-mix` (working title)
- Language: Persian (fa-IR)
- Intended use: train/evaluate a full-duplex speech-to-speech prototype
- Last updated: 2026-08-22

## Splits (status)

| Split | Source | License | Publish? | Status |
|---|---|---|---|---|
| Internal 100–200 h | 2TB S3 YouTube | YouTube ToS — research-internal | No raw audio | CSV long-form inventory **455.753 h**. Audio ingested **200.00 h** (219617 clips). Caption-filtered speech mix **197.613 h** / 215684 utterances (in 100–200 h band). It contains no assistant response supervision. |
| Synthetic duplex | tiled harmonic overlap mixer | synthetic | Yes, clearly labeled | **20.0 h**; plumbing and regression use only, not evidence of conversational speech quality. |
| Optional local audio | Lab/home interactions | consent | Only if separately approved | **not collected; not definition-required**. Default human-study mode retains lossy features and metrics but no WAV. |
| Pointers | Common Voice fa | CC-0 | Pointers only | Not downloaded here |

S2S/TTS text = YouTube **CSV caption** (`transcript_caption`) after **fa-verbatim-2**. Those CSVs are the same reference transcripts used to fine-tune Soroush; NeMo is not a teacher for this mix. ASR training orthography (`prepare_tabaghe16.normalise`) is not used as the spoken target. NeMo HTTP remains the cascade ASR baseline only. Optional `youtube_reasr.jsonl` (120.00 h) is diagnostic.

## Audit snapshot

`results/corpus_audit.json` is authoritative. The latest in-process audit reports:

- 215,684 rows / 197.613 h;
- 0 duplicate IDs, 0 schema errors, and 0 group leaks;
- 24 exact texts shared across splits;
- interruption labels: {'none': 215684};
- 0 overlap annotations and 0 response-supervision rows;
- thesis conversational coverage: **False**.
- caption/ASR screening proxy: 110,024 Digiato rows / 101.091 h, mean token similarity 0.7243, low-similarity fraction 0.0613;
- Zoomit live spot check: n=20, mean similarity 0.7735, low-similarity fraction 0.0.
Independent-ASR agreement is a screening proxy, not ground-truth alignment; retain manual audio review in the validation protocol.

The hours gate and conversational-supervision gate are separate; empty/default schema fields are not collected labels.

## Labels

The schema supports `utt_id`, `audio_path`, `duration`, `transcript_caption`, `transcript_nemo` (diagnostic), `speaker_id`, `overlap_intervals`, `interrupt_label` (`interrupt` / `backchannel` / `noise` / `none`), `snr`, `license`, and `age_bin`.

## Join rule

Where the Tabaghe16 layout holds, `{stem}_chunk_NNNN.wav` is 1:1 with CSV row N (1-indexed). Episodes with fewer than 50 rows are treated as Shorts and skipped.

Channels (2TB `asr`): Digiato, Zoomit, Kooshiar, Mehran Rowshan Persian — each `CSVs/` + `Audio_Chunks/`.

## Collection notes

Chunk audio is time-aligned to YouTube caption CSVs from the existing crawl (`--write-auto-subs` → VTT → CSV). SFT uses those captions, not a second ASR pass. Optional leftover `batch-reasr` / `youtube_reasr.jsonl` is not the training target. Diarization (Community-1) is optional on a podcast subset (`DIARIZATION_SERVICE_URL`).

Filter: duration 1–20 s, Persian script ratio, music-caption drop, SNR p90−p10, LID `language_status` when present. TTS/SFT uses the caption-filtered mix (`filtered.jsonl` / `filtered_caption.jsonl`), not `--require-teacher`.

## Ethical

All participants sign the mode-specific `docs/CONSENT.md` and explicitly opt into persistence. Feature and metrics modes retain no WAV. Public release excludes YouTube and restricted media. Human study N=5–10 (≥2 aged 60+) remains incomplete; see `docs/HUMAN_STUDY.md`.
