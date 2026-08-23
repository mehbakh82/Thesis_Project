# Duplex interaction prompt sheet

Use with `.venv/bin/python -m thesis_s2s.cli serve --study --retention features --session-id S001 --speaker-id P01 --age-bin 60plus`.
Written, mode-specific consent (`docs/CONSENT.md`) **before** any study persistence.

In feature mode no WAV is stored. Lossy aggregate features and labels go to `turns.jsonl`.

## Mix

- Quiet room (majority of hours)
- One household-noise pass (TV / dishes)
- Across the human study, include at least two participants aged 60+.

## Prompts (force each label)

| id | What the speaker does | `interrupt_label` |
|---|---|---|
| warmup_time | بپرس: «ساعت چند است؟» | none |
| interrupt_story | بخواه داستان بلند بگوید؛ **وسط پاسخ حرف بزن** | interrupt |
| backchannel | در حین پاسخ فقط «آها» / «بله» بگو — سیستم **نباید** قطع شود | backchannel |
| noise | با نویز پس‌زمینه یک سؤال بپرس | noise |
| free | سه دقیقه گفت‌وگوی آزاد | none / interrupt as it happens |

## After the session

```bash
.venv/bin/python -m thesis_s2s.cli export-recordings
```

Do not publicly release raw audio or derived participant data unless the approved consent and data-management plan explicitly permit it.
