# Barge-in detector v2 — result (acoustic classifier benchmark)

Protocol: `docs/BARGEIN_DETECTOR_V2_PROTOCOL.md` (with the balanced-accuracy
amendment and the labeling-closure amendment, both before test access).
Receipt (write-once): `results/eval/bargein_detector_v2_final.json`.

## Test (94 events, 31 sessions, opened once)

| Model | Accuracy | F1 | Balanced accuracy | False-accept rate |
|---|---:|---:|---:|---:|
| Chosen: gradient-boosted trees, threshold 0.775 | 0.574 (session CI 0.45–0.70) | 0.672 (0.54–0.79) | 0.590 | 0.381 |
| Energy rule | 0.755 | 0.859 | 0.503 | 0.952 |
| Secondary task, interrupt vs backchannel (51 events) | 0.529 | 0.636 | 0.535 | 0.455 |

Labels: Gemini 3.8 Flash, two independently worded passes, agreed events only
(independent automatic labels, not human-verified).

## Interpretation

**Negative.** On real podcast events, an energy/F0/MFCC classifier cannot
tell floor-taking speech from backchannels and other non-floor events much
better than chance (balanced accuracy 0.59). The energy rule's high F1 is an
artefact of the 3:1 class balance: it says "stop" for nearly everything (95%
false accepts). In a mono podcast window both voices share one channel, which
is harder than deployment, where the assistant's own voice is known; but the
result also shows that acoustic features alone cannot separate a backchannel
from an interruption. That is the reason the deployed controller confirms
lexically with the ASR (`docs/BARGEIN_SIMULATION_RESULT.md`: test accuracy
0.933, F1 0.880 in the deployment simulation; 0.85 / 0.85 in the real-browser
loopback).

The test set is small (94 events) because labeling stopped at the API budget
limit; intervals are wide.
