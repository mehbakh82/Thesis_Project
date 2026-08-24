# Dataset card (Persian speech corpus and planned duplex extension)

## Identity

- Name: `fa-s2s-duplex-mix` (working title)
- Language: Persian (fa-IR)
- Intended use: train/evaluate a full-duplex speech-to-speech prototype
- Last updated: 2026-08-24

## Splits (status)

| Split | Source | License | Publish? | Status |
|---|---|---|---|---|
| Internal conversation candidates | University S3 YouTube archive | Supervisor-approved internal thesis use; source licenses not independently verified | No raw audio | Full inventory **775.887 caption h**. Final deterministic plan: **196.546 candidate h / 296 episodes / 1021 windows / 219.403 staging h**; **181.824 automatic multi-speaker h** and **105.727 estimated response-pair h / 6017 pairs**. Authorization passes for all 1021 windows; 40-row window QA and 24-row/short-excerpt interaction QA remain pending. |
| Synthetic duplex | tiled harmonic overlap mixer | synthetic | Yes, clearly labeled | **20.0 h**; plumbing and regression use only, not evidence of conversational speech quality. |
| Optional local audio | Lab/home interactions | consent | Only if separately approved | **not_collected; not definition-required** (0.000 h, 0 turns, elderly_turns=0). Default study kit is `serve --study --retention features`; it stores no WAV. |
| Pointers | Common Voice fa | CC-0 | Pointers only | Not downloaded here |

The internal-use decision is recorded in `docs/SUPERVISOR_DECISIONS.md`; `results/conversation_source_authorization_report_combined.json` separately reports authorization for 1021 final staging windows and zero verified-license coverage. Redistribution remains disabled.

The active limited-training policy is the documented student QA waiver in
`docs/QA_WAIVER.md`; it is not supervisor approval of the waiver. The two QA
sheets remain 0/40 and 0/24 and are preserved for later review. Waiver outputs
must report zero human-verified rows/interruptions and false strict thesis
coverage.

S2S/TTS text = YouTube **CSV caption** (`transcript_caption`) after **fa-verbatim-2**. Those CSVs are the same reference transcripts used to fine-tune Soroush; NeMo is not a teacher for this mix. ASR training orthography (`prepare_tabaghe16.normalise`) is not used as the spoken target. NeMo HTTP remains the cascade ASR baseline only. Optional `youtube_reasr.jsonl` (120.00 h) is diagnostic.

## Audit snapshot

`results/prepared_episode_audit.json` reports the primary reconstruction: **287/287 episodes**, **947 windows / 201.576 h**, reconstruction gate **True**.
The bounded reserve audit reports **22 episodes / 182 windows / 43.156 h**; the deterministic selector uses 9 reserve episodes.

`results/diarized_episode_audit_combined_authorized.json`: **296 episodes / 1021 windows / 181.824 multi-speaker h / 217.115 aligned h**; automatic and authorization gates pass, while manual QA remains open.
`results/conversation_yield_estimate_combined.json`: **6017 estimated non-reused pairs / 105.727 pair h**. Labels: {"none": 1230, "overlap_unattributed": 4787}; human-verified direct interruption pairs: 0.
Raw speaker boundaries recover **712 conservative interaction candidates** ({"backchannel": 43, "interrupt": 669}). The generated 24-row sheet covers all four channels and about three minutes of excerpt audio. These are automatic candidates—not interruption claims—until pair-level listening review passes.
`results/conversation_audit.json` now verifies the built waiver-bound corpus:
**6017 pairs / 105.727 h / 149 sessions / 371 speaker IDs**, with zero missing
files, reused spans, or session-split leaks. The limited waiver gate passes;
strict coverage and every human-verification claim are false.

Under the waiver, these 712 candidates may remain training pseudo-label
metadata but none becomes a human-verified label. The strict audit fields remain
false even when the separate limited-training waiver field passes.

Staging hours preserve conversational context and gaps, whereas pair hours count only non-reused adjacent-turn spans. They are intentionally audited as different measures; the final pair set is inside the 100–200 h thesis band.

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

After either completed strict QA or validation of the explicit limited-training waiver, non-reused adjacent turns are exported in the official Moshi stereo schema. The user channel retains authorized natural archive audio. The primary assistant channel is synthesized deterministically from the approved next-turn text with the pinned Mana-Persian-Piper voice. Original podcast response audio is an explicit multi-voice ablation. Neither derivative corpus is redistributed; waiver-derived data is not represented as human-verified.
