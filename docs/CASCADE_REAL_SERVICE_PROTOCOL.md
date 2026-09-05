# Real-service cascade acceptance protocol

Status: frozen before execution. This is a positive working-system acceptance test, not a
generalization, human-quality, or official end-to-end latency experiment.

## Purpose

Demonstrate that the implemented fallback path executes the intended real components together:
the running Persian NeMo ASR service, locally cached `Qwen/Qwen2.5-0.5B-Instruct`, and the pinned
Persian Piper voice. Silent substitution by rules or formant synthesis is a failure.

## Frozen inputs and execution

- Input manifest: `data/processed/moshi_finetune_v6_overfit/train.jsonl`.
- The manifest must contain exactly 32 train-only rows. A path containing `test` is rejected.
- Fixed row indices: `0, 3, 7, 11, 15, 19, 23, 27, 31`.
- Each source hash must match its manifest entry.
- Each pair must be 16 kHz, signed 16-bit stereo; only channel 0 (the user channel) enters ASR.
- ASR endpoint: `http://127.0.0.1:8090`, with a 20-second request timeout.
- Responder: locally cached `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding, at most 64 new tokens.
- TTS: `models/piper/fa_IR-mana-medium.onnx`.
- Entrypoint: `.venv/bin/python scripts/evaluate_cascade_real_service.py`.
- Output is write-once: `results/eval/cascade_real_service_train_panel.json`.

The report binds the manifest, evaluator, protocol, Piper model, Piper configuration, and local
Qwen revision. It stores input hashes rather than source transcripts. The sealed final test is not
read or scored.

## Per-sample pass criteria

Every one of the nine samples must satisfy all of the following:

1. source audio hash matches the manifest;
2. ASR returns without an error and at least four characters;
3. at least 80% of transcript letters use the Arabic/Persian script;
4. the responder backend is exactly the requested Qwen model, initialization succeeded, and no
   deterministic rule fallback was used;
5. the reply has at least four characters and at least 80% of its letters use the Arabic/Persian
   script;
6. the TTS backend is exactly Piper, never the formant fallback;
7. reply audio is finite, normalized, at least 1,600 samples, and has RMS at least `0.001`.

The overall result passes only if ASR health is HTTP 200, Qwen is loaded, the panel is exact, and
all nine samples pass.

## Permitted claim and limitations

A passing result supports: “The real NeMo → Qwen → Piper Persian cascade produced valid Persian
text and non-silent speech for every item in a predeclared nine-item train-only acceptance panel.”

It does not establish held-out generalization, conversational naturalness, task success, an
official 12–24 GB deployment result, or the thesis browser `T_first_audio` target. Full-turn timing
is recorded only as descriptive engineering telemetry. These limitations must remain attached to
the result in the thesis and project documentation.
