# Data provenance and release register

Every dataset entering training or evaluation must have one row in this register or an equivalent machine-readable manifest. Training requires either a verified source license or documented internal-research authorization; the audit reports them separately.

| Source | Role | Access/license | Raw publish? | Current evidence |
|---|---|---|---|---|
| University S3 YouTube crawl | Persian acoustics/captions; conversation candidates | Supervisor-approved internal thesis training; source licenses not independently verified | No | Final measured plan: 296 episodes / 1,021 windows / 196.546 candidate h / 219.403 staging h; 181.824 automatic multi-speaker h and 6,017 estimated response pairs / 105.727 pair h. All 1,021 windows are internally authorized. The 40-row window QA and 24-row interaction QA are pending. Raw speaker boundaries produce 712 automatic candidates, but none counts as a verified interruption before listening review. |
| CALLFRIEND Farsi Second Edition (`LDC2014S01`) | Proposed natural telephone conversation | LDC institutional/member license required | No | Not acquired |
| MATERIAL Farsi-English (`LDC2024S13`) | Proposed natural conversation and varied environments | LDC institutional/member license required | No | Not acquired |
| Live participants | Usability, latency, interruption | Explicit mode-specific consent and ethics approval | No by default | Not collected |
| Mana-Persian-Piper synthetic assistant targets | Consistent system channel for Moshi; natural user audio and reference response text remain from the approved crawl | Model card MIT; Mana-TTS declared CC0; exact revision/weight hash pinned | No model weight or generated corpus | Local voice verified; deterministic synthesis/export implemented |
| Synthetic harmonic data | Unit/regression tests only | Project-generated | Yes with code-license decision | Present; not scientific speech evidence |

Authorization evidence is summarized in `docs/SUPERVISOR_DECISIONS.md` and machine-checked for the final plan in `results/conversation_source_authorization_report_combined.json`; neither record asserts an open source-data license.

Conversation listening QA is deliberately waived by the student for the active
limited run because it cannot be completed or delegated within the schedule;
this is separate from the supervisor-approved data-use basis. Every resulting
pair must carry the exact waiver hash, all interaction candidates remain
automatic pseudo-labels, and no human-verification or strict-coverage claim is
permitted. See `docs/QA_WAIVER.md`.

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
.venv/bin/python -m thesis_s2s.cli normalize-conversation-rights-metadata
.venv/bin/python -m thesis_s2s.cli audit-prepared-episodes
.venv/bin/python -m thesis_s2s.cli audit-corpus
.venv/bin/python -m thesis_s2s.cli audit-conversations
.venv/bin/python -m thesis_s2s.cli release-snapshot
```

For the limited path, pass the same
`--qa-waiver configs/conversation_qa_waiver.yaml` to pair building,
`audit-conversations`, and `export-moshi-data`. Omitting the flag selects the
strict path; supplying a missing, changed, invalid, or mismatched waiver fails
closed.

`audit-corpus` validates the current acoustic manifest. `audit-conversations` is the binding supervision audit for natural response pairs. Neither replaces legal review or human listening.

## Release exclusions

Do not include raw YouTube/LDC/participant audio, access credentials, internal object-store paths, or third-party model weights in a public archive. Publish aggregate statistics, code, schemas, hashes where permitted, and a reproducible acquisition/preprocessing description.
