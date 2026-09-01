# Moshi v6.1 deterministic-text decoding follow-up

Status: frozen after the completed v6 negative and before any v6.1 generation.

## Scope

The completed v6 runtime report at
`results/moshi_v6_overfit_runtime.json` (SHA-256
`fc3c394a76e8866baf40fe649a0ec5c78d0b857aa1307d3c2cbd9caf16961d6b`)
is an infrastructure-valid diagnostic negative. It produced audible, mostly
Persian output at later checkpoints but failed the unchanged all-nine gate. It
is preserved and is not reclassified by this follow-up.

## Single intervention

- Reuse the unchanged v6 checkpoints, training manifest, and nine-row panel.
- Keep the official no-fuse LoRA execution proven equivalent in v6.
- Keep audio sampling at the official defaults (`temperature=0.8`, top-k 250).
- Keep text temperature at 0.7 but set text top-k from 25 to 1, making only the
  text token choice deterministic.
- Do not train, select new rows, inspect a final test, or alter a quality gate.

The exact generation config is
`configs/moshi_v6_text_greedy_runtime.json`.

## Positive criterion

All four checkpoints (50, 100, 150, 200) are exercised once. A candidate is
positive only if every one of the same nine rows still passes every v6 gate:
complete prompt, official-server integrity, audible finite audio with RMS at
least `1e-3`, at least four normalized text characters, Persian-letter fraction
at least `0.80`, no Unicode replacement character, and target-prefix CER at
most `0.75`. The same text-loss rule applies: the earliest all-nine candidate
is selected only if its fixed-scope text loss is at least 30% below step 50,
unless step 50 itself passes.

## Decision

- Positive: deterministic text decoding resolves the v6 exposure failure;
  retain it as the deployment decoding policy, still without generalization.
- Negative: decoding alone is insufficient. Do not relax the gates; freeze a
  separate train-only scheduled-input/text-dropout experiment or pivot away
  from Moshika.
