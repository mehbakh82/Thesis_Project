# Recorded-audio interruption proxy protocol

Frozen: 2026-08-31, before extracting acoustic features, fitting the recorded
proxy detector, selecting its threshold, or evaluating its held-out test split.

## Scope and evidence class

This benchmark reuses the supervisor-approved internal-research YouTube
recordings. Its exact source is
`data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl`
with SHA-256
`b7afb3df52fcfadb91d639808db18382cfbfd006750891b2248a125393177440`.

The audio is real recorded Persian conversation, but the event labels are
automatic diarization/alignment proxies. No person listened to and verified
these events. The benchmark therefore measures consistency with the frozen
automatic heuristic and must always report `official_detector_eligible: false`.
It cannot satisfy the thesis's human-ground-truth >80% gate, regardless of its
numeric result. It is useful evidence about pipeline readiness and domain shift.

## Frozen events and split

A positive event is a conservative raw-diarizer boundary classified as
`interrupt`: distinct adjacent speakers, at least 0.5 raw/aligned turn match,
boundaries within 0.5 seconds, and at least 0.2 seconds overlap. Negative events
are short `backchannel` candidates or high-confidence distinct-speaker clean
turns with a raw gap from 0.2 to 2.0 seconds. The overlap/gap values and labels
are never model inputs.

Sessions—not clips—are assigned by
`sha256("20260831:" + session_id) mod 100`: buckets 0–69 train, 70–84
validation, and 85–99 test. At most eight events of each kind per session are
retained by stable event hash. Within every split, all retained backchannels are
kept first and stable-hash clean turns fill a negative set equal in size to the
interrupt set. The run fails closed unless every split contains both classes,
at least five sessions, and at least 20 events per class.

## Features, model, and threshold

For each event, only the 400 ms acoustic context ending 120 ms after the
candidate response onset contributes features. Features are energy, zero
crossing rate, YIN F0/voicing, 13 MFCCs and 13 deltas, aggregated by mean,
standard deviation, and maximum. No transcript, session, channel, timestamp,
speaker label, overlap duration, or gap is a model feature.

The existing fixed 200-tree GBDT configuration is trained once on train
sessions. Candidate probability thresholds 0.05–0.95 in 0.05 increments are
ranked only on validation sessions by balanced accuracy, then interruption F1,
then distance to 0.5, then lower threshold. The selected threshold is frozen
before the test probabilities are scored. The test split is evaluated once.

The report includes confusion matrices, accuracy, interruption F1, FAR, FRR,
event-level intervals, session-block bootstrap intervals, a fixed energy/ZCR
baseline, exact hashes, group-overlap checks, and class/channel counts. Raw
audio, event clips, and feature vectors are not copied or retained. The small
model file is local/ignored; its hash is recorded in the report.
