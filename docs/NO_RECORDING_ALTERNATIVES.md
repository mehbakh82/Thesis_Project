# No-raw-recording alternatives

The thesis definition does **not** state that 8–15 hours must be newly recorded. That number was an earlier engineering recommendation, not a binding requirement. The definition does require (a) 100–200 hours of conversational Persian with the relevant conditions and labels, and (b) a 5–10-person evaluation including at least two people aged 60+.

Two different constraints must therefore be separated.

## If live sessions are possible but raw voice cannot be retained

Use the implemented feature-retention mode:

```bash
.venv/bin/python -m thesis_s2s.cli serve --study --retention features \
  --session-id MOS01 --speaker-id P01 --age-bin under_60
```

The microphone is processed live, but no WAV is written. The system retains only:

- a lossy 400 ms aggregate of energy, zero-crossing rate, F0/voicing, MFCC, and delta-MFCC;
- the intended condition label and whether playback stopped;
- client timing, GPU identity/VRAM, age bin, pseudonymous speaker/session IDs, ratings, and optional notes.

Use `--retention metrics` if even acoustic aggregates are disallowed. That preserves timing, stop events, labels, and ratings, but cannot train the feature-based held-out detector. Consent is still required because study data are retained.

This path can satisfy the live usability and physical-GPU measurements without storing identifiable raw speech, subject to supervisor/ethics approval. Run every scripted condition for every participant so a speaker-held-out fold contains both interrupt and non-interrupt examples.

## If no live participant session is possible

Code cannot manufacture the required human-study evidence. The defensible choices are:

1. Ask the supervisor/lab to recruit or host 5–10 short live sessions, including two people aged 60+, with feature-only or metrics-only retention.
2. Run remote, supervised sessions in the browser with no raw-audio retention. The participant still speaks live; only approved telemetry is stored.
3. Ask for a written scope amendment: replace the completed interaction study with an approved listening study, expert evaluation, or a fully specified future-work protocol. This changes the requirement and must be documented as such.
4. If no amendment is approved, report the human study as incomplete. Synthetic users or self-entered scores must never be represented as participants.

A listening-only study can measure naturalness and preference but cannot, by itself, validate live interruption behavior or elderly turn-taking.

## Conversational-data alternatives that do not require new recording

The preferred route is now the verified internal YouTube archive:

- 775.887 caption-speech hours are available across 1,442 long episodes;
- a balanced 180.043-hour whole-episode candidate set is already selected from Tabaghe16, Mehran Rowshan Persian, Digiato, and Zoomit;
- the provided CSV captions supply text, while diarization supplies speaker timing;
- Kooshiar remains inventory/acoustic data and is excluded from the default conversation set.

This route avoids new recording, but it is not automatically requirement-complete. Confirm that internal YouTube use is permitted, run diarization/alignment, reject single-speaker windows, and manually review a stratified sample. The pre-cut chunks may lose cross-boundary overlap, which the manifests explicitly declare.

LDC is now a fallback if licensing, audio quality, or verified multi-speaker yield leaves fewer than 100 hours:

- [CALLFRIEND Farsi Second Edition Speech](https://catalog.ldc.upenn.edu/LDC2014S01): about 42 hours of two-channel natural telephone conversations.
- [MATERIAL Farsi-English Language Pack](https://catalog.ldc.upenn.edu/LDC2024S13): about 61 hours of conversational telephone speech, with speakers aged 16–67 and varied environments; only part is transcribed.

The implemented commands are:

```bash
.venv/bin/python -m thesis_s2s.cli plan-conversation-corpus
.venv/bin/python -m thesis_s2s.cli prepare-conversation-episodes
.venv/bin/python -m thesis_s2s.cli audit-prepared-episodes
.venv/bin/python -m thesis_s2s.cli diarize-conversation-episodes
.venv/bin/python -m thesis_s2s.cli audit-diarized-episodes
.venv/bin/python -m thesis_s2s.cli sample-conversation-qa
.venv/bin/python -m thesis_s2s.cli apply-conversation-qa
.venv/bin/python -m thesis_s2s.cli create-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli apply-conversation-rights-review
.venv/bin/python -m thesis_s2s.cli build-conversations \
  --in-jsonl data/processed/manifests/conversation_episode_windows_approved.jsonl
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli export-omni2-data
```

The conversation audit fails closed unless hours, response audio/text, multi-speaker provenance, licenses, interruption/backchannel coverage, a manual-verification sample, and source files are present.

## Recommended decision

Use licensed/archive conversational speech for the 100–200-hour corpus and run only short, no-WAV live sessions for the human evaluation. If even short live sessions are impossible, obtain a written scope amendment before treating any substitute as requirement-complete.
