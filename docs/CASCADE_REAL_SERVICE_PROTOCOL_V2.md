# Real-service cascade acceptance protocol v2

Status: frozen before v2 execution. This remains a positive working-system acceptance test, not
a generalization, human-quality, or official end-to-end latency experiment.

## Versioned correction

The immutable v1 result
`results/eval/cascade_real_service_train_panel.json`
(SHA-256 `6aa0a29131d7741537b84b4277572e8d683891a97a408325eb459c38a3370fe7`)
failed. Its telemetry established two implementation defects:

1. the installed Transformers tokenizer returned a `BatchEncoding`, while the runtime passed it
   directly to `generate()` instead of extracting `input_ids` and `attention_mask`; all nine
   turns therefore used the explicit rule fallback;
2. Piper output could overshoot `[-1, 1]` slightly during resampling.

The runtime now extracts the tensors, forwards the attention mask, and clips Piper float output at
the public synthesis boundary. Regression tests cover both changes. No data, fixed indices,
language thresholds, component requirements, or audio thresholds were relaxed. The v2 report
must preserve and hash-bind the v1 failure and the repaired runtime sources.

## Frozen inputs and execution

- Input manifest: `data/processed/moshi_finetune_v6_overfit/train.jsonl`.
- The manifest must contain exactly 32 train-only rows. A path containing `test` is rejected.
- Fixed row indices: `0, 3, 7, 11, 15, 19, 23, 27, 31`.
- Each source hash must match its manifest entry.
- Each pair must be 16 kHz, signed 16-bit stereo; only channel 0 enters ASR.
- ASR endpoint: `http://127.0.0.1:8090`, with a 20-second request timeout.
- Responder: cached `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding, at most 64 new tokens.
- TTS: `models/piper/fa_IR-mana-medium.onnx`.
- Entrypoint: `.venv/bin/python scripts/evaluate_cascade_real_service_v2.py`.
- Write-once output: `results/eval/cascade_real_service_train_panel_v2.json`.
- Source transcripts and generated reply strings are omitted; their hashes and script statistics
  are retained.

The sealed final test is not read or scored.

## Unchanged per-sample criteria

Every one of the nine samples must satisfy all criteria:

1. matching source hash;
2. error-free ASR, at least four transcript characters, and at least 80% Persian/Arabic-script
   letters;
3. exact Qwen backend, successful initialization, successful generation, and no rule fallback;
4. at least four reply characters and at least 80% Persian/Arabic-script letters;
5. exact Piper backend;
6. finite normalized reply audio with at least 1,600 samples and RMS at least `0.001`.

Overall passage additionally requires ASR HTTP health 200, the exact nine-row panel, the bound v1
failure, and no final-test access.

## Permitted claim and limitations

A pass supports only: “The real NeMo → Qwen → Piper Persian cascade produced valid Persian text
and non-silent speech for every item in a predeclared nine-item train-only acceptance panel.”

It does not establish held-out generalization, conversational naturalness, task success, an
official 12–24 GB deployment result, or browser `T_first_audio`. Full-turn timing is descriptive
engineering telemetry only.
