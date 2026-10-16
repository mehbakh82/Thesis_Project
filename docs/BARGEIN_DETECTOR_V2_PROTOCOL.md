# Barge-in detector v2 — protocol

Status: frozen before any v2 model was trained; the test sessions are opened
once, after labeling is complete.

## Why a v2

The v1 real-audio proxy (81.06% accuracy, 78.99% F1) used labels produced by
the same diarizer-boundary rules for training, validation, and test; its
positives were overlaps and its negatives silent turn gaps, so a loudness rule
scored 78%, and only five backchannels were tested. The live server also loaded
the synthetic sine-wave model instead. v2 replaces the labels, the task
framing, and the evaluation.

## Labels (independent of the candidate rules)

Candidate events are mined from diarized windows of all four corpus channels
(`scripts/mine_interaction_events.py`): overlap onsets, short responses, and
turn changes. A stratified sample (at most five per session and candidate kind;
2,782 events, 272 sessions) is cut into 4.5 s clips centred on the onset
(`scripts/extract_event_clips.py`). Gemini 3.8 Flash labels each clip twice
with independently worded prompts (`scripts/label_interaction_events.py`) as
`interrupt`, `backchannel`, `turn_change`, `noise`, or `none`; only events on
which both passes agree are used. These are independent automatic labels, not
human-verified labels. Two-pass agreement is reported.

## Task

- **Primary (deployment decision):** floor-taking speech (`interrupt` or
  `turn_change`) versus non-floor events (`backchannel`, `noise`, `none`) — i.e.
  "should the assistant stop?".
- **Secondary (hardest confusion):** `interrupt` versus `backchannel` only.

The decision window runs from 100 ms before to 480 ms after the onset (Gemini's
onset estimate when both passes place it within ±0.5 s of the diarizer onset),
matching the live controller's confirmation horizon.

## Features and models

Per 20 ms frame: log energy, ZCR, YIN F0, voicing, 13 MFCCs and deltas
(`src/thesis_s2s/bargein/features.py`); per window: mean, standard deviation,
maximum, first-half and second-half means, voiced fraction, energy slope, and
energy range. Candidates: an energy rule (share of post-onset frames above
−35 dBFS), logistic regression, gradient-boosted trees, and the proposal's MLP
(64-32 units, early stopping). Training examples are augmented (two extra
copies each) with point-source noise at 5–20 dB SNR, simulated room responses,
and simulated residual assistant echo (Piper speech at −25 to −10 dB); validation
and test audio are not augmented.

## Splits and selection

Sessions are split by `sha256("detector-v2-20261016:" + session) mod 100`:
test < 15, validation 15–29, training ≥ 30, so no session spans splits. The
`develop` stage selects, on validation only, the model with the highest F1 and
its threshold (grid 0.20–0.80). The `final` stage refits that frozen choice on
training + validation and evaluates the test sessions once.

## Amendment before test access (2026-10-16)

The first development run (validation only; test untouched) showed that
selecting by validation F1 is degenerate under the roughly 3:1 positive/negative
imbalance: the best-F1 thresholds made every candidate answer "stop" for nearly
every event (false-accept rates 0.66–0.78), and an always-stop rule already has
F1 ≈ 0.85. Model and threshold are therefore selected by **balanced accuracy** (mean of
recall on floor-taking and non-floor events). Because the 15-session
validation split is small, selection uses session-grouped 5-fold
cross-validation over the training and validation sessions together (pooled
out-of-fold predictions; augmentation only inside training folds). Speaker-
contrast features (voiced-frame means of the decision window minus those of a
900 ms pre-onset reference, MFCC distance, F0 and energy change) were added
because a mono podcast window always contains both voices. Accuracy and F1
remain the reported target metrics; the test sessions were not opened before
this amendment.

## Reported metrics and target

Accuracy, F1, precision, recall, false-accept/false-reject rates, balanced
accuracy, confusion counts, and session-bootstrap 95% intervals, for the chosen
model and the energy rule, plus the secondary task. The proposal target is
interrupt-detection accuracy ≥ 80% (F1); this project requires both test
accuracy and test F1 above 0.80 to call the target met on these labels.

## Live controller (separate measurement)

The deployed two-stage controller (`src/thesis_s2s/runtime/bargein_live.py`)
ducks the assistant after two voiced 32 ms frames and then stops or resumes.
Its onset-to-duck and onset-to-stop latencies and its stop/resume decisions are
measured with the browser loopback harness on scripted interruptions and
backchannels; that is reported separately from the classifier test above.
