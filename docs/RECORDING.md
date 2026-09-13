# Optional raw-audio collection protocol

The thesis definition does not prescribe an 8–15-hour new-recording minimum. Raw-audio collection is optional unless the supervisor explicitly requires it. Prefer `docs/HUMAN_STUDY.md` feature-only mode when voice cannot be retained.

Written Audio-mode consent (`docs/CONSENT.md`) **before** any persistence. Prompt sheet: `docs/PROMPT_SHEET.md`.

```bash
export PYTHONPATH=src
python3 -m thesis_s2s.cli serve --record --study --session-id S001 --speaker-id P01 --age-bin 60plus
# then: python3 -m thesis_s2s.cli export-recordings
```

Open the local browser client and obtain explicit written and on-screen consent. Use a new path-safe `session_id` for each session and a stable pseudonymous `speaker_id` across that speaker's sessions.

The server accepts persistence consent only as the JSON boolean `true`; it does
not coerce strings or numbers into consent. It also rejects unsafe identifiers,
unknown prompt/label/age values, and malformed ratings before persistence.

- The browser streams continuously during user collection and assistant playback; do not mute the microphone while the system talks.
- After an interruption stops playback, continue the same utterance and press the green button once to end it. The rolling microphone pre-roll and subsequent speech form the next user turn.
- 16 kHz WAV is stored under `data/recordings/<session_id>/` with `turns.jsonl` containing client and server timing separately.
- Labels are `interrupt` / `backchannel` / `noise` / `none`; annotate overlap intervals and response pairs during export/curation.
- Record `age_bin`, environment/noise condition, microphone, exact model versions, and `license=consent`.
- Mix: quiet room + one household-noise pass
- If this optional set is collected, include diverse speakers and conditions; the binding age requirement remains at least two participants aged 60+ in the human study.
- Split train/validation/test by speaker or session, never by individual clip.
- Review every public row for consent and identifiers. YouTube stays internal; synthetic audio must remain clearly identified.

Run `.venv/bin/python -m thesis_s2s.cli study-summary` after collection. A recording directory or partial rating is not evidence that the study is complete.
