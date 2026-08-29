# Moshi v2 complete-prompt validation correction

Frozen: 2026-08-29, after the superseded validation selector failed and before
any corrected candidate generation or v2 final-test access.

## Verdict on the superseded panel

The 2,000-step v2 training retry completed successfully. All 20 checkpoints
passed exact artifact validation, and their total complete-scope validation
loss decreased from 2.312407 at step 100 to 1.734574 at step 2,000.

The first post-training official-server panel was nevertheless invalid as a
response-generation eligibility test. Its evaluator always streamed only the
first five seconds of the user channel. Eight of the nine frozen rows contained
user speech beyond that boundary:

| validation index | last non-zero user PCM (s) | complete in 5 s |
|---:|---:|:---:|
| 0 | 2.8288 | yes |
| 16 | 57.1480 | no |
| 32 | 442.1130 | no |
| 48 | 109.0780 | no |
| 64 | 68.0870 | no |
| 80 | 41.9990 | no |
| 96 | 102.3200 | no |
| 112 | 15.6410 | no |
| 130 | 16.8010 | no |

A duplex model should normally remain silent while the user is still speaking.
Consequently, requiring a response to each truncated prefix confounded
turn-taking with response capability and could not support checkpoint
eligibility. This is an evaluation-input defect, not a threshold failure.

The generated evidence is preserved without reinterpretation:

- `results/moshi_v2_runtime_candidates_truncated_prompt_invalid.json`
- `results/moshi_v2_checkpoint_selection_truncated_prompt_failed.json`
- `results/hardware/moshi_v2_posttraining_truncated_prompt_failed.json`

That receipt records `test_access_started=false`. The final-test objective and
runtime outputs did not exist when this correction was frozen.

## Corrected input-only panel rule

No candidate, loss, model output, or test row is used to construct the
replacement panel.

1. Read the unchanged ordered 131-row validation manifest.
2. For each stereo PCM16 WAV, find the last non-zero sample in channel 1, the
   user channel.
3. Retain a row only when that endpoint is at or before the unchanged
   five-second streaming boundary.
4. From the eligible manifest-ordered rows, choose nine floor-spaced ranks:
   `floor(i * (eligible_count - 1) / 8)`, for `i = 0..8`.

There are 17 eligible validation rows. The deterministic corrected indices are
`[0, 11, 33, 51, 55, 74, 85, 87, 106]`. Their user-channel endpoints are all
at or before 4.960 seconds.

The candidate set remains steps 100 through 2,000. The exact artifact checks,
complete-scope validation losses, official-server requirement, RMS threshold
0.001, nonempty text requirement, Persian-letter fraction threshold 0.5, and
9/9 panel requirement are unchanged. Among eligible candidates, selection
still uses minimum complete-validation total loss with an earlier-step exact
tie break.

This is transparently a post-training correction to validation inputs. The
selection criterion and thresholds were predeclared before training; the exact
corrected panel indices were not. Both the superseded and corrected outcomes
must be reported.

## Untouched final-test rule

Checkpoint selection must finish before any v2 final-test access. If selection
passes, the one-time objective test remains unchanged. The separate automatic
runtime diagnostic applies the same input-only complete-prompt rule to the
frozen final-test manifest after selection; it selects nine floor-spaced rows
from the eligible complete-prompt pool without using model outputs. It is not
used to select or revise the checkpoint.

If the corrected validation panel yields no eligible candidate, v2 remains a
negative result and the pipeline must stop before the test set. Thresholds must
not be weakened after seeing the corrected outputs.
