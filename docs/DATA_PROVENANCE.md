# Data provenance and release register

Every dataset entering training or evaluation must have one row in this register or an equivalent machine-readable manifest. Unknown licenses fail the conversation audit.

| Source | Role | Access/license | Raw publish? | Current evidence |
|---|---|---|---|---|
| University S3 YouTube crawl | Persian acoustics/captions; possible interview subset | Internal; source-specific rights review required | No | 197.613 h filtered, but current set lacks response/overlap supervision |
| CALLFRIEND Farsi Second Edition (`LDC2014S01`) | Proposed natural telephone conversation | LDC institutional/member license required | No | Not acquired |
| MATERIAL Farsi-English (`LDC2024S13`) | Proposed natural conversation and varied environments | LDC institutional/member license required | No | Not acquired |
| Live participants | Usability, latency, interruption | Explicit mode-specific consent and ethics approval | No by default | Not collected |
| Synthetic harmonic data | Unit/regression tests only | Project-generated | Yes with code-license decision | Present; not scientific speech evidence |

## Required per-source fields

- immutable source ID and acquisition date;
- owner/provider and source URL/catalog ID;
- exact license or approved internal-use basis;
- media and annotation hashes where contractually permitted;
- language, channel, speaker/session, duration, and collection conditions;
- transcript/diarization/overlap annotation source;
- human-verification status and sampling method;
- allowed training, evaluation, derived-weight, thesis, and release uses;
- deletion/retention date for participant data;
- preprocessing code commit and output-manifest checksum.

## Split and annotation rules

- Split natural conversations by source session before clipping.
- Keep every speaker/session group in exactly one split.
- Do not inflate hours by reusing the same source interval in multiple response pairs.
- Preserve original text and automatic text separately.
- Mark pseudo-labels as automatic until manually reviewed.
- Report natural, read, synthetic, and participant evidence separately.

## Machine checks

```bash
.venv/bin/python -m thesis_s2s.cli audit-corpus
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli release-snapshot
```

`audit-corpus` validates the current acoustic manifest. `audit-conversations` is the binding supervision audit for natural response pairs. Neither replaces legal review or human listening.

## Release exclusions

Do not include raw YouTube/LDC/participant audio, access credentials, internal object-store paths, or third-party model weights in a public archive. Publish aggregate statistics, code, schemas, hashes where permitted, and a reproducible acquisition/preprocessing description.
