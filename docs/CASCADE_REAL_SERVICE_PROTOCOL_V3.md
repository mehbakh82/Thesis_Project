# Real-service cascade acceptance protocol v3

Status: frozen before v3 execution. This is a working-system acceptance test, not a held-out
generalization, human-quality, or official end-to-end latency experiment.

## Versioned correction

The immutable v2 result
`results/eval/cascade_real_service_train_panel_v2.json`
(SHA-256 `77e8e03af947659539efd650eb2e79c7e2a49a3a7bb55aeca5be91187e75bdbf`)
passed every gate on eight of nine rows. The remaining real Qwen response contained a Python code
block and scored 73% Persian-script letters against the predeclared 80% threshold.

During development, the failed train-only row was used to test a stricter system instruction. The
runtime instruction now explicitly requires Persian script and prohibits Latin characters or
words. This is disclosed train-panel iteration: it improves the production response constraint but
does not create held-out evidence. No data, panel index, model, decoder, component, threshold, or
acceptance criterion changed. v3 must hash-bind the v2 result and exact runtime sources.

## Frozen execution

- Manifest: `data/processed/moshi_finetune_v6_overfit/train.jsonl`, exactly 32 train-only rows.
- Fixed indices: `0, 3, 7, 11, 15, 19, 23, 27, 31`.
- Input: hash-matching 16 kHz signed 16-bit stereo pairs; only channel 0 enters ASR.
- ASR: `http://127.0.0.1:8090`, 20-second timeout.
- Responder: cached `Qwen/Qwen2.5-0.5B-Instruct`, greedy, at most 64 new tokens.
- TTS: `models/piper/fa_IR-mana-medium.onnx`.
- Entrypoint: `.venv/bin/python scripts/evaluate_cascade_real_service_v3.py`.
- Write-once output: `results/eval/cascade_real_service_train_panel_v3.json`.
- The evaluator rejects a test-named manifest and omits source transcripts and reply strings.

## Unchanged criteria

All nine rows must have: a matching input hash; error-free ASR; at least four transcript
characters; at least 80% Persian/Arabic-script transcript letters; exact initialized Qwen backend;
successful Qwen generation with no rule fallback; at least four reply characters; at least 80%
Persian/Arabic-script reply letters; exact Piper backend; and finite normalized non-silent reply
audio of at least 1,600 samples and RMS at least `0.001`.

Overall passage also requires ASR HTTP health 200, the exact panel, the hash-bound v2 parent, and no
sealed-final-test access.

## Permitted claim

A pass supports only: “The real NeMo → Qwen → Piper Persian cascade produced valid Persian text
and non-silent speech for every item in a predeclared nine-item train-only acceptance panel.”

It does not establish held-out generalization, naturalness, semantic task success, an official
12–24 GB deployment result, or browser `T_first_audio`. Full-turn timing is descriptive only.
