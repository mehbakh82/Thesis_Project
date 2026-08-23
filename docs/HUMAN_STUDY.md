# Human study protocol

**Status: incomplete until recruitment and validation are finished.** This document is the session script. N=5–10 native Persian speakers, **at least two aged 60+**. Recording kit: `docs/RECORDING.md` + `docs/PROMPT_SHEET.md`.

```bash
export PYTHONPATH=src
.venv/bin/python -m thesis_s2s.cli serve --study --retention features --session-id MOS01 --speaker-id E01 --age-bin 60plus
```

Open the displayed local URL, read the mode-specific consent form with the participant, and require the participant to check the on-screen consent box. An unchecked turn is processed transiently but is not persisted.

`features` is the default study mode and writes no WAV. Use `metrics` if acoustic aggregates are prohibited, or `audio`/`--record` only with explicit permission for raw voice. See `docs/NO_RECORDING_ALTERNATIVES.md`.

## Session (about 25 minutes)

1. Consent (`docs/CONSENT.md`).
2. Two warm-up turns with the duplex client.
3. Scripted tasks: ask the time, interrupt a long reply, produce a backchannel (`آها`, `بله`) that should **not** stop the system, speak in a slightly noisy room if possible.
4. Free conversation (3 minutes).
5. Ratings (`docs/MOS_SHEET.md`). Compare energy-VAD vs GBDT if time allows (`--detector energy` vs default `gbdt`).

## What to log

- Client-observed `t_first_audio_ms` from `playback_started`
- Client-observed `t_barge_in_ms` from `playback_stopped_ack`
- Server generation and detection times as separate diagnostic fields
- Whether each intended interrupt actually stopped playback
- False stops on backchannels or coughs
- Qualitative notes for elderly users (pause length, loudness, turn-taking)
- All four 1–5 rating fields and free-text notes

After every collection block, run:

```bash
.venv/bin/python -m thesis_s2s.cli export-recordings
.venv/bin/python -m thesis_s2s.cli study-summary
```

Do not report a complete study unless `results/eval/human_study.json` confirms 5–10 participants, at least two aged 60+, complete ratings, client timing, a real speaker/session-held-out detector result above the target, and eligible physical-GPU evidence.

## Success for the thesis chapter

Report MOS, live interrupt success, Likert satisfaction, confidence intervals, and elderly-specific failure modes. Formant fallback audio is a development aid and must never enter MOS; record the exact Piper/neural TTS model and checksum used for every rated session.

The software reports p50, p95, and maximum latency. Do not choose the binding 500 ms statistic until the supervisor clarifies the definition.
