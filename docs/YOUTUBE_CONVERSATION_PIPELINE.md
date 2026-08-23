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
  -> automatic diarization on the 4090
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

Provide credentials through the `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`,
`S3_ENDPOINT`, and `S3_BUCKET=asr` environment variables. Never place them
in the repository or command arguments.

```bash
# Fast: CSV-only inventory and deterministic selection.
.venv/bin/python -m thesis_s2s.cli plan-conversation-corpus

# Long I/O job: resumable, one episode at a time, bounded 15-minute windows.
.venv/bin/python -m thesis_s2s.cli prepare-conversation-episodes

# GPU job using the existing offline Community-1 container.
diar_ip=$(docker inspect asr_nemo_soroush_diarization \
  --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')
curl -f "http://${diar_ip}:8081/health/ready"
export DIARIZATION_SERVICE_URL="http://${diar_ip}:8081"
.venv/bin/python -m thesis_s2s.cli diarize-conversation-episodes
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes

# Create a deterministic channel/pass-reject sample. A reviewer listens to the
# referenced internal clips, fills the CSV, then applies the decisions.
.venv/bin/python -m thesis_s2s.cli sample-conversation-qa
# Edit data/processed/manifests/conversation_manual_qa.csv
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa

# Only verified/aligned windows from the reviewed manifest enter this builder.
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli export-omni2-data
```

Use `--max-episodes 1` for an I/O smoke test and
`diarize-conversation-episodes --limit 1` for a service smoke test.
If verified primary yield is too low, add a bounded reserve batch:

```bash
.venv/bin/python -m thesis_s2s.cli prepare-conversation-episodes \
  --selection data/processed/manifests/youtube_conversation_reserve.jsonl \
  --max-source-hours 20
```


The completed real-data preparation smoke test downloaded one Mehran episode:
296/296 caption chunks decoded, two windows, 0.307 reconstructed audio hours,
and no errors.
A diagnostic H100 service test (not official 4090 evidence) completed both
windows and aligned 82.85% and 87.72% of their captions. Both were correctly
rejected as multi-speaker evidence: the secondary speaker occupied only 8.201
seconds/1.34% and 3.915 seconds/2.63%, respectively. This confirms both the live
service contract and the need for episode-level filtering plus a reserve pool.


## Storage and runtime expectations

180 hours of 16-kHz mono PCM16 output is about 20.7 GB before response-pair
exports. The source transfer is larger because the stored chunks can use less
compact encodings and the selected episodes are downloaded before
reconstruction. The preparation command deletes only its own temporary
per-episode downloads and is resumable.

The 4090 is most useful for the diarization stage, followed by any approved
model adaptation and live evaluation. Diarization is performed on bounded
windows so an entire multi-hour podcast is never sent as one request.

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

Manual QA means listening to and checking existing archive clips; it does not
require the student to record new speech. A supervisor, lab member, or paid
annotator can perform it under the approved data-access rules. The required
sample size and acceptance thresholds remain supervisor decisions. The
`sample-conversation-qa` command samples every available channel × automatic
pass/reject stratum deterministically; `apply-conversation-qa` fails incomplete
or incorrect checks closed and never marks licensing as approved. See
`SUPERVISOR_QUESTIONS_FA.md` and `MANUAL_QA_FA.md`.
