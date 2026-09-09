# YouTube conversation-corpus pipeline

This document is the operational contract for turning the existing YouTube
crawl into conversation candidates. It deliberately separates candidate hours
from verified conversational supervision.

## Verified inventory

The inventory was generated on 2026-08-23 from the provided CSV files in the
2-TB MinIO bucket. An episode is long-form eligible when it has at least 50 CSV
caption rows.

| Source | Prior | Eligible episodes | Caption speech hours | Skipped short CSVs |
|---|---|---:|---:|---:|
| Tabaghe16 | interview podcast | 195 | 320.134 | 185 |
| Mehran Rowshan Persian | interview podcast | 185 | 87.308 | 1,645 |
| Digiato | mixed technology speakers | 414 | 148.086 | 548 |
| Zoomit | mixed technology speakers | 524 | 132.535 | 1,334 |
| Kooshiar | primarily monologue | 124 | 87.824 | 8 |
| **Total** | | **1,442** | **775.887** | **3,720** |

These are sums of caption-segment durations, not a claim of 775.887 hours of
conversation. Shorts, uncaptioned time, music, sponsor sections, and gaps
explain why this differs from informal estimates such as “about 900 hours.”

## Current 180-hour plan

The deterministic weighted-fair selector chose 287 whole episodes and
180.043 caption hours:

| Source | Episodes | Hours | Share |
|---|---:|---:|---:|
| Tabaghe16 | 56 | 92.469 | 51.36% |
| Mehran Rowshan Persian | 108 | 50.999 | 28.33% |
| Digiato | 50 | 18.119 | 10.06% |
| Zoomit | 73 | 18.456 | 10.25% |

Kooshiar is excluded by default. It is eligible only when a title explicitly
indicates an interview, guest, conversation, or podcast; no episode met that
conservative rule in this selection. The inventory still retains all Kooshiar
episodes for acoustic or monologue experiments.
A disjoint reserve contains 172 additional episodes and 120.327 candidate
hours. It is not part of the 180.043-hour primary plan and must be prepared only
in small batches if diarization leaves fewer than 100 verified hours.


The selector operates over the complete pool, preserves every selected episode
as one train/validation/test group, caps a source at 55% of the target, and
never treats a channel prior as a speaker label.

## Data flow

```text
all five CSV pools
  -> long-form episode inventory
  -> weighted whole-episode candidate selection (180 h)
  -> download one episode's ordered chunks at a time
  -> reconstruct <=15-minute windows with CSV-to-window time mapping
  -> automatic diarization on the H100
  -> maximum-overlap assignment of each CSV segment to a speaker
  -> reject low-confidence and single-speaker windows
  -> create non-reused adjacent user/response pairs
  -> audit hours, grouping, overlap, labels, license, files, and manual QA
```

## How the CSV references and NeMo are used

- The provided YouTube CSV text is the canonical transcript target for this
  corpus. Every reconstructed segment retains its original row number, source
  time, raw caption, and normalized caption.
- Diarization supplies speaker and overlap timing. It does not replace the CSV
  text. A caption is usable only when at least 50% of its duration overlaps one
  diarized speaker; a window is alignment-complete only at 80% usable coverage.
- A window is an automatic multi-speaker candidate only when it has at least
  two speakers, at least 20 seconds from the second speaker, at least a 3%
  second-speaker share, and complete reference alignment.
- The NeMo ASR model remains the live cascade ASR and an optional diagnostic
  cross-check. Its hypothesis must not silently overwrite the provided CSV
  reference. A manual audit may compare NeMo and CSV text to find bad captions.

## Important chunk limitation

The verified 2-TB paths contain caption CSVs and pre-cut audio chunks. The
pipeline reconstructs ordered windows and preserves short source gaps, but it
does not claim that these are untouched original episodes.

Pre-existing chunk boundaries may remove an interruption or overlap that
crossed a boundary. Every window therefore carries:

- `is_original_episode_audio=false`;
- `cross_chunk_overlap_recoverable=false`;
- an explicit overlap-limitation note.

Within-chunk overlap can still be detected. Cross-boundary overlap is never
fabricated. If untouched episode audio becomes available and its use is
permitted, it should replace reconstructed windows for overlap experiments.

## Commands

Prefer a protected preconfigured rclone remote when available:

```bash
export THESIS_RCLONE_REMOTE=your_remote_name
```

The code validates the remote name and maps internal `:s3:` paths without
reading, copying, or printing credentials from rclone configuration. The
alternative is ephemeral `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`,
`S3_ENDPOINT`, and `S3_BUCKET=asr` environment variables. Never place secrets
in the repository or command arguments.

```bash
# Fast: CSV-only inventory and deterministic selection.
.venv/bin/python -m thesis_s2s.cli plan-conversation-corpus

# Long I/O job: resumable, one episode at a time, bounded 15-minute windows.
.venv/bin/python -m thesis_s2s.cli prepare-conversation-episodes
.venv/bin/python -m thesis_s2s.cli normalize-conversation-rights-metadata
.venv/bin/python -m thesis_s2s.cli audit-prepared-episodes

# GPU job using the existing offline Community-1 container.
diar_ip=$(docker inspect asr_nemo_soroush_diarization \
  --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
curl -f "http://${diar_ip}:8081/health/ready"
export DIARIZATION_SERVICE_URL="http://${diar_ip}:8081"
export DIARIZATION_TIMEOUT_SECONDS=600
.venv/bin/python -m thesis_s2s.cli diarize-conversation-episodes
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes
.venv/bin/python -m thesis_s2s.cli annotate-conversation-noise
.venv/bin/python -m thesis_s2s.cli estimate-conversation-yield

# The non-mutating estimate decides whether reserve is needed; it is not a
# training-ready audit. After the final merge, apply the recorded internal-use
# decision to a derived manifest. License and authorization remain separate.
.venv/bin/python -m thesis_s2s.cli apply-conversation-rights-review \
  --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl \
  --report results/conversation_source_authorization_report_combined.json

# A reviewer listens to the referenced archive clips; no recording is needed.
.venv/bin/python -m thesis_s2s.cli sample-conversation-qa \
  --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl
# Edit data/processed/manifests/conversation_manual_qa.csv and the already
# generated 24-row conversation_interruption_qa.csv.
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa \
  --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli apply-interruption-qa \
  --in-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes \
  --manifest data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl \
  --min-hours 100 --max-hours 240

# Only reviewed, aligned, and training-authorized windows enter this builder.
# An explicitly reviewed failure is always excluded.
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
# Primary direct-model export: natural user audio and reference response text,
# rendered in one pinned Persian assistant voice.
.venv/bin/python -m thesis_s2s.cli export-moshi-data --assistant-audio-mode piper
# Optional multi-voice ablation only:
.venv/bin/python -m thesis_s2s.cli export-moshi-data \
  --assistant-audio-mode source \
  --out-dir data/processed/moshi_finetune_source_ablation \
  --report results/moshi_source_ablation_report.json
```

Use `--max-episodes 1` for an I/O smoke test and
`diarize-conversation-episodes --limit 1` for a service smoke test.
If verified primary yield is too low, add a bounded reserve batch:

```bash
.venv/bin/python -m thesis_s2s.cli prepare-conversation-episodes \
  --selection data/processed/manifests/youtube_conversation_reserve.jsonl \
  --out-root data/processed/conversation_reserve_windows \
  --out-jsonl data/processed/manifests/conversation_reserve_episode_windows.jsonl \
  --max-source-hours 20

.venv/bin/python -m thesis_s2s.cli diarize-conversation-episodes \
  --in-jsonl data/processed/manifests/conversation_reserve_episode_windows.jsonl \
  --out-jsonl data/processed/manifests/conversation_reserve_episode_windows_diarized.jsonl
.venv/bin/python -m thesis_s2s.cli annotate-conversation-noise \
  --in-jsonl data/processed/manifests/conversation_reserve_episode_windows_diarized.jsonl \
  --out-jsonl data/processed/manifests/conversation_reserve_episode_windows_noise_labeled.jsonl

# Use the minimum whole-episode reserve prefix needed for a 123.5 h source-pair
# safety target while keeping selected candidate hours at or below 220.
.venv/bin/python -m thesis_s2s.cli select-conversation-reserve

.venv/bin/python -m thesis_s2s.cli merge-conversation-windows \
  --inputs \
    data/processed/manifests/conversation_episode_windows_noise_labeled.jsonl \
    data/processed/manifests/conversation_reserve_tabaghe16_windows_selected.jsonl \
  --out-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined.jsonl

.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes \
  --manifest data/processed/manifests/conversation_episode_windows_noise_labeled_combined.jsonl \
  --out results/diarized_episode_audit_combined.json \
  --min-hours 100 --max-hours 250
```

Keep primary and reserve preparation/diarization manifests separate. Alternate
jobs now receive distinct progress-report paths. The merge command writes
atomically and rejects every duplicate `window_id`; the yield selector records
input/output hashes and stops on whole-episode boundaries.

Measured final state:

- primary: 287 episodes, 947 windows, 201.576 staging h, 164.247 automatic multi-speaker h, 199.288 aligned h, zero diarization failures;
- audited reserve pool: 22 episodes, 182 windows, 43.156 staging h, 42.907 automatic multi-speaker h, zero failures;
- selected reserve: all 22 prepared episodes / 182 windows, 39.903 candidate h and 28.744 estimated pair h;
- final plan: 219.946 candidate h, 309 episodes, 1,129 windows, 244.733 staging h, 207.154 automatic multi-speaker h, 242.445 aligned h, and 6,754 response pairs / 123.796 source-pair h;
- final derivative: 6,754/6,754 pairs / 108.584 measured stereo h, split 6,419/131/204, zero independent-audit failures, and 24/24 deterministic assistant re-synthesis matches;
- authorization: all 1,129 windows approved for internal thesis training, zero source-license-verified hours, redistribution disabled;
- noise evidence: primary 639 clean / 243 moderate / 59 noisy / 6 unestimated windows plus reserve 157 clean / 25 moderate windows;
- interaction evidence: 5,411 pairs remain conservatively `overlap_unattributed`; raw speaker boundaries expose 770 stricter automatic candidates (717 interruption-like / 53 backchannel-like) across all four channels, with zero human-verified direct interruptions under the waiver.

### Balanced v2 (post-training recommended corpus)

The frozen corpus above remains the input attributed to completed Moshi runs.
Balanced v2 is a separate, later lineage: 105 disjoint Digiato/Zoomit episodes
with at least 95% local chunk coverage were prepared and diarized, then combined
with the frozen authorized windows using the opt-in
`max_duration_interval_v2` adjacent-turn selector. The two pair manifests are
merged with hash receipts and duplicate/split-leak rejection before a
deterministic session-round-robin cap is applied to Tabaghe16:

```bash
.venv/bin/python -m thesis_s2s.cli estimate-conversation-yield \
  --manifest data/processed/manifests/conversation_balance_v2_extension_windows_diarized.jsonl \
  --pair-selection-method max_duration_interval_v2 \
  --out results/conversation_balance_v2_extension_yield.json

.venv/bin/python -m thesis_s2s.cli merge-conversation-pairs \
  --inputs \
    data/processed/manifests/conversation_balance_v2_pairs_all_initial.jsonl \
    data/processed/manifests/conversation_balance_v2_pairs_extension.jsonl \
  --out-jsonl data/processed/manifests/conversation_balance_v2_pairs_all.jsonl \
  --report results/conversation_balance_v2_pair_merge.json

.venv/bin/python -m thesis_s2s.cli select-balanced-conversations \
  --manifest data/processed/manifests/conversation_balance_v2_pairs_all.jsonl \
  --out-jsonl data/processed/manifests/conversations_balanced_v2.jsonl \
  --report results/conversation_balance_v2_selection.json \
  --target-max-channel-share 0.54 --min-hours 100 --max-hours 200

.venv/bin/python -m thesis_s2s.cli audit-conversation-balance \
  --manifest data/processed/manifests/conversations_balanced_v2.jsonl \
  --out results/conversation_balance_v2_audit.json \
  --max-channel-share 0.55 --min-channels 4

.venv/bin/python -m thesis_s2s.cli audit-conversations \
  --manifest data/processed/manifests/conversations_balanced_v2.jsonl \
  --out results/conversation_balanced_v2_audit.json \
  --qa-waiver configs/conversation_qa_waiver.yaml
```

The complete pool has 8,787 pairs / 164.830 h and no duplicate IDs or
cross-split session leaks. The final selection has 6,551 pairs / 110.374 h / 186
sessions: Digiato 17.921 h, Mehran Rowshan Persian 24.800 h, Tabaghe16 59.602 h,
and Zoomit 8.050 h. Every minority-source pair is retained, the dominant share
is 54.0%, and the structural audit reports 0 missing files, 0 reused spans, 0
split leaks, and 6,551/6,551 authorized/common-waiver rows. These remain
automatic labels; v2 does not complete listening QA or verified-interruption
requirements and was not used by the completed model runs.

The 40-row window QA handoff covers five windows from every channel × automatic
pass/reject stratum and balances clean, moderate, noisy, and unestimated
conditions. A second deterministic sheet sampled six interaction candidates
per channel (24 rows, 168.3 seconds of excerpt audio) from the earlier
712-candidate pool. Both are preserved strict-path artifacts but are explicitly
waived—not completed—for the active limited run; neither requires new recording.


## Storage and runtime expectations

180 hours of 16-kHz mono PCM16 output is about 20.7 GB before response-pair
exports. The source transfer is larger because the stored chunks can use less
compact encodings and the selected episodes are downloaded before
reconstruction. The preparation command deletes only its own temporary
per-episode downloads and is resumable.

The prepared-audio audit verifies that selected caption hours remain in the
100–200 planning band, reconstructed staging audio stays below a 250-hour operational cap,
and windows have unique IDs, preserved episode splits, ordered reference rows,
bounded durations, valid WAV format, matching manifest/file durations, and a
current—not stale—completion report. Only final verified multi-speaker hours
are subject to the strict thesis maximum of 200.
The report also records SHA-256 checksums for the selection manifest, prepared
window manifest, and completion statistics.

The H100 is the correct machine for diarization and approved model adaptation.
The 4090 is reserved for the final physical 24 GB fit and live-browser latency
run. Diarization is performed on bounded windows so an entire multi-hour
podcast is never sent as one request.
The primary diarizer writes
`data/processed/manifests/conversation_episode_diarization_stats.json` after
every attempted window. Alternate outputs derive distinct `*_stats.json`
filenames, so reserve jobs cannot overwrite primary evidence. Resumed runs
retain aggregate verified/aligned hours and never repeat completed GPU requests.


## Evidence gates and manual QA

`results/conversation_selection_audit.json` currently passes the planning
gate and fails the evidence gate, as intended. Candidate hours cannot become a
thesis-compliance claim until:

1. 100–200 hours of windows pass automatic multi-speaker and reference
   alignment checks;
2. response-pair audio and text are actually exported without source-interval
   reuse or episode split leakage;
3. overlap/interruption/backchannel coverage is measured honestly;
4. licensing for internal training and thesis reporting is confirmed; and
5. a stratified sample of diarization, speaker assignment, captions, and
   interaction labels is manually reviewed.

For the current limited run, item 5 is explicitly waived by the student because
the review cannot be completed or delegated within the available schedule. The
automatic-only path is hash-bound by `configs/conversation_qa_waiver.yaml` and
may satisfy only `training_ready_under_qa_waiver`; it cannot satisfy strict
thesis coverage or create human-verified interruption evidence. The blank
sheets, Persian instructions, and playback helper are preserved for later use.
See `docs/QA_WAIVER.md` for commands, permitted claims, thesis disclosure, and
the strict recovery path.

Manual QA means listening to and checking existing archive clips; it does not
require the student to record new speech. A supervisor, lab member, or paid
annotator can perform it under the approved data-access rules. The required
sample size and acceptance thresholds remain supervisor decisions. The
`sample-conversation-qa` command samples every available channel × automatic
pass/reject stratum deterministically and balances noise conditions within each stratum; `apply-conversation-qa` fails incomplete
or incorrect checks closed and never marks licensing as approved. See
`SUPERVISOR_QUESTIONS_FA.md` and `MANUAL_QA_FA.md`.
