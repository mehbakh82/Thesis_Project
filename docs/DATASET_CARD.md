# Dataset card (Persian speech corpus and planned duplex extension)

## Identity

- Name: `fa-s2s-duplex-mix` (working title)
- Language: Persian (fa-IR)
- Intended use: train/evaluate a full-duplex speech-to-speech prototype
- Last updated: 2026-08-23

## Splits (status)

| Split | Source | License | Publish? | Status |
|---|---|---|---|---|
| Internal conversation candidates | University S3 YouTube archive | Supervisor-approved internal thesis use; source licenses not independently verified | No raw audio | Full inventory **775.887 caption h**. Final deterministic plan: **196.546 candidate h / 296 episodes / 1,021 windows / 219.403 staging h**; H100 processing found **181.824 automatic multi-speaker h** and **105.727 estimated response-pair h / 6,017 pairs**. Authorization passes; 40-row listening QA remains pending. |
| Synthetic duplex | tiled harmonic overlap mixer | synthetic | Yes, clearly labeled | **20.0 h**; plumbing and regression use only, not evidence of conversational speech quality. |
| Optional local audio | Lab/home interactions | consent | Only if separately approved | **not_collected; not definition-required** (0.000 h, 0 turns, elderly_turns=0). Default study kit is `serve --study --retention features`; it stores no WAV. |
| Pointers | Common Voice fa | CC-0 | Pointers only | Not downloaded here |

The internal-use decision is recorded in `docs/SUPERVISOR_DECISIONS.md`; `results/conversation_source_authorization_report_combined.json` separately reports authorization for all 1,021 final staging windows and zero verified-license coverage. Redistribution remains disabled.

S2S/TTS text = YouTube **CSV caption** (`transcript_caption`) after **fa-verbatim-2**. Those CSVs are the same reference transcripts used to fine-tune Soroush; NeMo is not a teacher for this mix. ASR training orthography (`prepare_tabaghe16.normalise`) is not used as the spoken target. NeMo HTTP remains the cascade ASR baseline only. Optional `youtube_reasr.jsonl` (120.00 h) is diagnostic.

## Audit snapshot

`results/prepared_episode_audit.json` is authoritative for the primary
reconstruction: **287/287 episodes**, **947 windows**, **201.576 h**, and zero
schema/split/order/ID/file/WAV/duration/failure errors. The bounded reserve audit
also passes for **22/22 episodes / 182 windows / 43.156 h**. The deterministic
yield selector then uses only nine reserve episodes; its hashes and stopping
proof are in `results/conversation_reserve_yield_selection.json`.

The final automatic evidence is split deliberately:

- `results/diarized_episode_audit_combined_authorized.json`: **296 episodes / 1,021 windows / 181.824 multi-speaker h / 217.115 aligned staging h**, all internally authorized; only manual QA fails under the explicitly relaxed 100–240 h staging contract;
- `results/conversation_yield_estimate_combined.json`: **6,017 pairs / 105.727 pair h** within the 100–200 h final-pair band;
- `results/conversation_selection_audit.json`: the original episode-level planning audit. Planning passes; it is not final training evidence.

The staging and final-pair hour contracts are reported separately because the
reconstructed windows preserve context and gaps. The supervisor must confirm
which measurement is binding for the definition. The older flat-clip audit
below remains acoustic/caption evidence only.

The older flat-clip acoustic/caption audit in `results/corpus_audit.json` reports:

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

The schema supports `utt_id`, `audio_path`, `duration`, `transcript_caption`, `transcript_nemo` (diagnostic), `speaker_id`, `overlap_intervals`, `interrupt_label` (`interrupt` / `backchannel` / `overlap_unattributed` / `noise` / `none`), `snr`, `license`, and `age_bin`. `overlap_unattributed` explicitly does not count as interruption evidence.

## Join rule

Where the Tabaghe16 layout holds, `{stem}_chunk_NNNN.wav` is 1:1 with CSV row N (1-indexed). Episodes with fewer than 50 rows are treated as Shorts and skipped.

Channels (2TB `asr`): Digiato, Zoomit, Kooshiar, and Mehran Rowshan Persian use top-level prefixes. Tabaghe16 uses `STT/YT_PodCast_Chunks/{CSVs,Audio_Chunks}/طبقه 16`.

## Collection notes

Chunk audio is time-aligned to YouTube caption CSVs from the existing crawl (`--write-auto-subs` → VTT → CSV). SFT uses those captions, not a second ASR pass. Optional leftover `batch-reasr` / `youtube_reasr.jsonl` is not the training target. Diarization is required before selected episodes count as conversational supervision. Reconstructed windows declare that cross-chunk overlap may be unrecoverable.

Filter: duration 1–20 s, Persian script ratio, music-caption drop, SNR p90−p10, LID `language_status` when present. TTS/SFT uses the caption-filtered mix (`filtered.jsonl` / `filtered_caption.jsonl`), not `--require-teacher`.

## Ethical

All participants sign the mode-specific `docs/CONSENT.md` and explicitly opt into persistence in the client. Feature and metrics modes retain no WAV. Public release excludes YouTube and restricted media. Human study N=5–10 (≥2 aged 60+) remains incomplete; see `docs/HUMAN_STUDY.md`.

## Moshi direct-model derivative

After diarization, deterministic speech/nonspeech RMS assigns an automatic background-noise condition that is included in stratified listening QA. After QA and rights application, non-reused adjacent
turns are exported in the official Moshi stereo schema. The user channel keeps
the natural archive audio. The primary assistant channel is deterministically
synthesized from the approved reference response text with the pinned
MIT-licensed Mana-Persian-Piper voice (revision and SHA-256 in
`third_party/UPSTREAMS.lock.json`). This follows Moshi's consistent-system-voice
training design. Original podcast response audio is retained only as an explicit
multi-voice ablation. Neither derivative is redistributed.
