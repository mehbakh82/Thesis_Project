# Dataset card (Persian speech corpus and planned duplex extension)

## Identity

- Name: `fa-s2s-duplex-mix` (working title)
- Language: Persian (fa-IR)
- Intended use: train/evaluate a full-duplex speech-to-speech prototype
- Last updated: 2026-08-23

## Splits (status)

| Split | Source | License | Publish? | Status |
|---|---|---|---|---|
| Internal conversation candidates | 2TB S3 YouTube | Source-specific internal use pending written confirmation | No raw audio | Full inventory **775.887 caption h**; plan **180.043 caption h / 287 episodes**; completed reconstruction **201.576 staging h / 947 windows / 287 episodes**. Prepared audit passes; conversation, rights, and manual-QA gates remain pending. |
| Synthetic duplex | tiled harmonic overlap mixer | synthetic | Yes, clearly labeled | **20.0 h**; plumbing and regression use only, not evidence of conversational speech quality. |
| Optional local audio | Lab/home interactions | consent | Only if separately approved | **not_collected; not definition-required** (0.000 h, 0 turns, elderly_turns=0). Default study kit is `serve --study --retention features`; it stores no WAV. |
| Pointers | Common Voice fa | CC-0 | Pointers only | Not downloaded here |

S2S/TTS text = YouTube **CSV caption** (`transcript_caption`) after **fa-verbatim-2**. Those CSVs are the same reference transcripts used to fine-tune Soroush; NeMo is not a teacher for this mix. ASR training orthography (`prepare_tabaghe16.normalise`) is not used as the spoken target. NeMo HTTP remains the cascade ASR baseline only. Optional `youtube_reasr.jsonl` (120.00 h) is diagnostic.

## Audit snapshot

`results/prepared_episode_audit.json` is authoritative for reconstructed
staging audio: **287/287 episodes**, **947 windows**, **201.576 h**, zero
schema/split/order/ID/file/WAV/duration/failure errors, and an explicit pending
rights marker on every row. It records SHA-256 hashes for all three source
manifests/reports. These are not yet verified multi-speaker hours.

`results/conversation_selection_audit.json` is authoritative for the episode-level plan. Planning gate: **True**; thesis evidence gate: **False**. The older flat-clip audit below remains acoustic/caption evidence only.

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

Channels (2TB `asr`): Digiato, Zoomit, Kooshiar, and Mehran Rowshan Persian use top-level prefixes. Tabaghe16 uses `STT/YT_PodCast_Chunks/{CSVs,Audio_Chunks}/طبقه 16`.

## Collection notes

Chunk audio is time-aligned to YouTube caption CSVs from the existing crawl (`--write-auto-subs` → VTT → CSV). SFT uses those captions, not a second ASR pass. Optional leftover `batch-reasr` / `youtube_reasr.jsonl` is not the training target. Diarization is required before selected episodes count as conversational supervision. Reconstructed windows declare that cross-chunk overlap may be unrecoverable.

Filter: duration 1–20 s, Persian script ratio, music-caption drop, SNR p90−p10, LID `language_status` when present. TTS/SFT uses the caption-filtered mix (`filtered.jsonl` / `filtered_caption.jsonl`), not `--require-teacher`.

## Ethical

All participants sign the mode-specific `docs/CONSENT.md` and explicitly opt into persistence in the client. Feature and metrics modes retain no WAV. Public release excludes YouTube and restricted media. Human study N=5–10 (≥2 aged 60+) remains incomplete; see `docs/HUMAN_STUDY.md`.
