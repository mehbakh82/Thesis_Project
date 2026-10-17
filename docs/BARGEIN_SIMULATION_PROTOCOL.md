# Barge-in deployment simulation — protocol

Status: written after three development runs (dev1–dev3, training-split audio
only) and before the lexical-confirm development run and the one-time final
run. The test split has not been used.

## Purpose

The detector v2 test (`docs/BARGEIN_DETECTOR_V2_PROTOCOL.md`) scores a
classifier on podcast windows, where the "user" and the other speaker are both
in one mono mix. The deployed situation is different: the microphone hears the
user plus a residual echo of the assistant's own voice. This simulation
measures the **deployed controller** (`src/thesis_s2s/runtime/bargein_live.py`,
Silero VAD, 20 ms frames) in that situation, built from real recordings
(`scripts/simulate_bargein.py`).

## Scenarios

Each scenario is the microphone signal during one assistant reply (Mana Piper
rendering of a training-split assistant turn, 4 s or longer):

- the reply itself as residual echo after the browser's echo cancellation,
  −32 to −18 dB relative to the user, delayed 20–60 ms after the played
  reference (the client reports the playback level to the server);
- background noise (OpenSLR point-source noises) −38 to −24 dB relative;
- one event at an early, middle, or late point of the reply, half of them
  convolved with an OpenSLR room response:
  - `interrupt`: a real single-speaker Persian turn of 1.2–4 s (positive);
  - `backchannel`: a real isolated listener token (short exclusive turn
    flanked by the other speaker, transcript only listener tokens);
  - `noise`: a real non-speech burst of 0.2–1.2 s, −6 to +3 dB relative;
  - `none`: nothing.

Speech, backchannels, noises, and replies are split by session or file
(`sha256("bargein-sim-20261016:" + key) mod 100 < 35` → test). Development
uses only training-split audio (300 fitting and 300 development scenarios,
equal classes and timings); the final run uses 600 test scenarios.

## Development so far (training split only)

- dev1–dev2: duration-only and MLP-assisted confirmation, echo-floor margins.
- dev3: added the double-talk detector against the client playback level.
  Best: accuracy 0.687, F1 0.624; stop rates interrupt 0.96, backchannel 0.56,
  noise 0.47, none 0.21. Duration/acoustic confirmation cannot separate long
  backchannels, loud noises, and leftover echo from interruptions.

## Amendment: lexical confirmation (before the final run)

A `lexical` confirm stage is added to the controller and the server: after the
duck, the deployed ASR (NeMo v4 CTC, the same service the cascade uses)
transcribes the captured speech since the onset at 400 ms of voiced speech,
every 200 ms after that while undecided, and when the user falls silent
(320 ms). Words of the reply being played (residual echo, matched exactly or
within one edit for words of four or more letters) and listener tokens
(`src/thesis_s2s/bargein/lexicon.py`) are discarded; any remaining word means
**stop**, while an empty or listener-token-only transcript at silence or at
1.6 s of speech means **resume**. ASR time is added to the decision time.

**Independence of the backchannel labels.** Backchannels are defined by the
same listener-token lexicon, so the clips that define the class must not be
selected by the confirming ASR. The backchannel pool is therefore re-mined with
a different model, SeamlessM4T v2 (v5 fine-tune), served through the same
HTTP contract; the simulator refuses to run lexical variants if the pool's
selection ASR is the confirm ASR. Speech, noise, and reply pools are unchanged.

Development variants: the dev3 duration/MLP grid (for comparison) plus
`lexical` with echo-floor margin 12 dB and double-talk margin ∈ {none, 6, 10,
14} dB.

### dev4 result and a stopped final start (2026-10-17)

dev4 (training split, Seamless-selected backchannel pool) chose
`lexical_margin_12.0_dtd_14.0`: accuracy 0.840, F1 0.765; stop rates interrupt
0.96, backchannel 0.25, noise 0.11, none 0.27 (duration-only with the same
double-talk margin: 0.797 / 0.719). An automated chain then started the final
run; it was killed about 20 seconds later, before it wrote any output or
printed any result, so that the remaining echo-only and backchannel false
stops could be analysed on development scenarios first. No test-split result
has been observed. The next development round and the final run follow the
rule below.

### dev5 (written before its results)

A diagnostic rerun of the dev4 development scenarios logged the transcript
behind every wrong stop. Three causes were found: (1) many false ducks happen
in the first 0.1–0.4 s of a reply, because the double-talk detector initialised
its echo-coupling estimate from the first playback frame, before the delayed
echo arrived; (2) echo words cut at the edges of the captured window
(«کانورسی…», «هیچوق…») escape the one-edit match; (3) lone function words
(«از», «این», «یه», «چه») counted as interruption content. dev5 reruns the same
300 development scenarios with three lexical families, each at double-talk
margins {none, 6, 10, 14} dB:

- `lexical`: the dev4 design (reproduction);
- `lexicalinit`: (1) fixed by a conservative coupling prior of 0 dB that
  adapts down while the user is quiet;
- `lexical2`: all three — the prior, echo matching that also accepts a
  prefix/suffix of three or more letters and one edit for three-letter words,
  and a closed-class function-word list (`FUNCTION_WORDS`) that neither
  confirms nor denies.

The duration/MLP grid is not rerun (dev4 already compared it). The variant
with the highest development accuracy, then F1, goes to the final run.

## Selection and final run

The development variant with the highest accuracy (then F1) is frozen and run
once on the 600 test scenarios (`--stage final`; the receipt is write-once).

## Reported metrics and target

Accuracy, F1, precision, recall (interrupt = positive, every other event
negative), confusion counts, stop rate per event kind and per timing, false
ducks before the event, onset-to-duck and onset-to-stop latency (median, p90,
p95, maximum, share ≤ 150 ms). The proposal asks for interruption detection
≥ 80% and reaction within 150 ms: the accuracy/F1 target is met only if both
exceed 0.80 on the test scenarios; the 150 ms criterion applies to the duck
(the "detect and adapt" reaction), and the stop latency is reported separately.

## Claim boundary

Simulated mixtures of real recordings, not live users; residual echo is
modelled as attenuated, delayed assistant speech; labels for speech and noise
come from the mining rules, for backchannels from an ASR listener plus the
lexicon. Not human-verified.
