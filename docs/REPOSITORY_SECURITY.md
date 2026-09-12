# Repository security and reproducibility review

Status date: 2026-09-12

## Controls that are enforced

- The GitHub repository is private and the configured remote is
  `https://github.com/mehbakh82/Thesis_Project.git`.
- Every reachable commit is authored and committed by
  `Mehran Bakhtiari <94431009+mehbakh82@users.noreply.github.com>`.
- The planning document, Persian project-definition document, raw/processed
  media, credentials, environments, model weights, and checkpoints are blocked
  by `.gitignore` and by the fail-closed history audit.
- `scripts/ci_artifact_audit.py` parses every tracked JSON/JSONL/YAML file,
  checks the project TOML, enforces a 5 MiB tracked-file ceiling, rejects
  forbidden paths and weight/secret suffixes, rejects private workstation
  paths, scans current text for high-confidence credentials, and scans every
  reachable Git revision for the same credential classes and forbidden paths.
  It performs the history scan locally and does not pass a repository token to
  a third-party action.
- Frozen candidate results remain bound to their exact pre-portability evaluator
  and protocol blobs. `scripts/verify_frozen_evaluator_sources.py` retrieves
  those blobs from their full commit ID, verifies their SHA-256 values and every
  declared receipt binding, and separately pins the current path-portable files.
- The proposal-alignment receipt binds the ignored private proposal's reviewed
  SHA-256 and the exact aggregate-evidence bytes/schema. CI fails if the public
  alignment summary drifts from `results/eval/EVIDENCE_STATUS.json`.
- GitHub Actions has read-only repository permissions, fetches complete history,
  and pins the first-party checkout and Python setup actions to immutable commit
  SHAs.
- GitHub dependency vulnerability alerts and automated security fixes are
  enabled; the authenticated API reports alerts available and automated fixes
  `enabled=true`, `paused=false`.
- CI compiles and lints `src`, `tests`, and `scripts`; mypy checks all 51
  source files against the Python 3.10 target.
- The measured branch-aware coverage is 80.32%; the enforced floor is 79%,
  raised from 35%, 60%, 68%, 71%, 74%, 75%, 76%, 77%, and 78% while retaining a small
  non-flaky margin. The floor is not raised to 80% because the remaining
  cross-interpreter margin is deliberately kept non-brittle.
  The CLI-exposed batch re-ASR pipeline is covered at 78%, multi-channel
  ingestion at 94%, YouTube preparation at 96%, the evaluation benchmark
  runner at 100%, and the historical codec/component survey at 57% after
  focused correctness tests. The data-factory orchestrator is covered at 89%
  after atomic-copy and stale-output regressions. Core conversation,
  Moshi-export, QA-policy, rights, and preflight modules are substantially
  above the aggregate; filtering and quality validation are at 74% and 76%.
  Recorded-proxy discovery, detector training, and the core detector are at
  64%, 70%, and 93% after focused integrity regressions.
- Future detector ingestion and evaluation fail closed on malformed JSONL,
  missing referenced audio, corrupt/non-finite features, unknown labels,
  absent speaker/session grouping, invalid threshold candidates, and invalid
  grouped-bootstrap inputs. Clean-turn discovery evaluates every adjacent
  speaker boundary, and the proposal's accuracy criterion is implemented as
  strictly greater than 80%, not greater than or equal to it. Detector reports
  produced by the training entry points explicitly identify their automatic
  metadata labels, zero human-verified labels, and official ineligibility.
  These controls apply to future runs; they do not regenerate or reclassify the
  frozen 81.06% automatic-label proxy.
- Detector fitting no longer truncates mismatched frame labels or accepts an
  empty, non-finite, wrong-width, non-integral, out-of-schema, or single-class
  training set. Empty audio deterministically produces no interruption, model
  replacement happens only after a successful fit, and prediction/evaluation
  inputs are schema-checked. Fitted detector files are written with fsync plus
  atomic replacement and a versioned payload; loading validates the fitted
  sklearn pipeline and feature width while retaining read-only compatibility
  with both existing trusted local legacy models. Pickle loading remains
  explicitly restricted to project-created trusted files.
- Every inventory/ingestion rclone subprocess has a positive finite configurable
  timeout (one hour by default). Environment-backed endpoint, bucket, provider,
  and named-remote values are validated before use; endpoints containing URL
  credentials, queries, or fragments are rejected. Inventory reports are
  status-labelled, finite JSON written with fsync and atomic replacement, and
  incomplete listings exit nonzero while retaining only error type/return code,
  never raw remote stderr. S3 inventory coverage is 87% after these regressions.
- Runtime ASR now rejects secret-bearing/malformed base URLs, invalid timeouts,
  empty request bodies, non-object/malformed/oversized responses, and invalid
  audio/sample rates. ASR failure no longer invents a greeting transcript in
  the component-latency helper; that rule-responder path is explicitly
  unofficial. Cached default Qwen loading is bound to the exact validated
  revision, prompt-v2 replies reject Latin text and over-25-word output, live
  client errors are redacted, and `/health` separately reports transport
  liveness and exact Qwen-v2/Piper readiness. The generic interrupt benchmark
  preserves upstream official ineligibility for automatic recorded labels
  instead of promoting any recorded fold. Cascade coverage is 88%.
- Audio/TTS boundaries reject invalid sample rates, unsupported/non-finite audio,
  ambiguous channels-first tensors, empty decodes, and empty WAV writes. WAV
  replacement is atomic and fsync-backed; ffmpeg and Piper subprocesses have
  positive finite configurable timeouts. Float input is clipped rather than
  heuristically reinterpreted as integer PCM, and the diagnostic formant fallback
  is stable across processes. Explicit missing Piper paths and arbitrary ONNX
  voices fail closed, output is schema/finite-checked, and deployment readiness
  requires a successful render probe rather than file presence. The complete
  suite passes 339 tests; audio and TTS coverage are 86.79% and 81.54%.
- Metric inputs now reject non-finite/non-integral/coerced binary labels,
  invalid percentile/bootstrap settings, non-numeric or negative latency/VRAM,
  and caller attempts to mark hardware-ineligible samples as official. Metric
  JSON rejects non-finite values and is fsync/atomically replaced without
  corrupting an existing report on failure. The complete suite passes 343
  tests, and metrics coverage is 83.85%.
- The legacy Whisper/projector reconstruction branch remains an explicitly
  gated ablation, not a deployable S2S system. It now uses restricted
  `weights_only` checkpoint loading, treats every direct-runtime declaration as
  ineligible until a real direct runtime exists, redacts load errors, rejects
  missing explicit manifests/audio instead of synthesizing replacements, and
  rejects its unimplemented Encodec option. Synthetic fixtures remain confined
  to the named smoke-test command. The complete suite passes 346 tests.
- CI-critical full-history and frozen-source Git commands have a validated,
  configurable 120-second timeout, while the pinned browser-client container
  build has a separate 30-minute bound. The evidence summary is fsync/atomically
  replaced and the privacy audit deterministically rejects any drift between
  `EVIDENCE_STATUS.json` and its generated Markdown view. The proposal and
  frozen-source receipts reproduce byte-for-byte. The complete suite passes 351
  tests, and evidence aggregation coverage is 97.35%.
- Consented study sessions now bind participant, retention, GPU identity,
  physical VRAM, and memory-cap provenance for the life of a session. Turn and
  rating JSONL appends are inter-process locked, finite-JSON checked, fsynced,
  and atomically replaced; a failed turn commit removes newly written audio.
  Manifest and study-summary replacement preserves the prior artifact on
  failure, API responses no longer disclose absolute recording paths, and
  malformed/non-finite stored turns fail the official study gate. The complete
  local suite passes 358 tests at 80.323% branch-aware coverage; the session
  recorder is covered at 90.35%.
- The general requirements audit remains fail-closed on unreviewed drift. On
  2026-09-09, pip-audit began reporting CVE-2026-69112 in Accelerate 1.14.0.
  The scanner feed later stopped mapping it and the resolver advanced to 1.15.0.
  No fixed release is declared, and direct inspection of the official 1.15.0
  wheel found the same unvalidated `weight_map` path join. The narrow policy
  pins 1.15.0, preserves the known risk whether or not the scanner emits it,
  and records the reviewed wheel hash. Any added finding, version change,
  missing resolved package, or policy drift fails CI. Trusted, pinned local
  model inputs remain mandatory. See
  `configs/application_dependency_risk_policy.json` and
  `results/security/application_dependency_audit.json`; reproduce it with
  `python scripts/audit_application_dependencies.py`.
- Critical Moshi assets, upstream revisions, and the isolated training
  environment are independently pinned and hash-checked by
  `requirements-moshi.lock`, `third_party/UPSTREAMS.lock.json`, and the
  hardware/environment evidence.
- The exact Qwen responder revision is pinned against the positive cascade
  report, all nine upstream names/revisions must appear in
  `THIRD_PARTY_NOTICES.md`, and citation/notice consistency is enforced by
  regression tests. The installed optional Piper runtime's GPL-3.0-or-later
  terms are explicitly separated from the Persian voice/model/data terms.
- The pinned Moshi browser source is built with
  `scripts/build_moshi_client.py` from a hash-verified dependency overlay in a
  digest-pinned Node 20 container. Its served production graph has zero known
  advisories; the static bundle's 33 files and tree hash are recorded in
  `results/hardware/moshi_client_build.json`. Eight remaining npm findings
  (five high, three moderate, zero critical) belong only to build-time tooling,
  which receives pinned trusted inputs in an ephemeral container. Any critical
  finding fails the build, and the npm development server is prohibited.
- Post-finalization storage cleanup removed only ignored/regenerable assets and
  non-promoted negative checkpoint tensors after their hashes and derived
  evidence were committed. Thirteen representative v1–v5 adapters remain
  hash-verified, alongside four separately retained v6.2 diagnostic checkpoints;
  the receipt uses no private absolute host path and is included in the release
  snapshot and authoritative evidence aggregation.
- The complete GitHub Actions run `34013864466` passed at duplex-ready commit
  `da6fed8` on 2026-09-06, including compile, lint, types, privacy/history,
  dependency, test/coverage, and pinned-scientific-risk jobs. The subsequent
  prior remote release run `34027190028` passed every gate at `edae181`. The
  subsequent post-intelligibility local audit at `6b4a7e9` passed all 182 tests,
  65% branch coverage, the 362-file history/privacy scan, general dependency
  audit, exact scientific-risk baseline, and all nine upstream pins. The final
  CI-portable pre-license evidence freeze at `fe32e6a` then passed the complete
  GitHub Actions run `34029248788`. A clean fresh full-history clone of that
  exact remote commit independently passed Ruff, mypy on 49 source files, the
  362-file tracked-artifact/history/privacy audit, and all 182 tests while the
  ignored private validation manifest was absent, as intended. The subsequent
  Persian-reporting freeze at `a0c4fcc` passed GitHub Actions run
  `34042698214`, including Ruff, mypy, 183 tests, coverage, all privacy and
  dependency gates, and the scientific-risk drift audit. A fresh full-history
  clone independently reproduced its exact remote HEAD, sole Mehran Bakhtiari
  author/committer identity, 363-file tracked audit, 310-file release snapshot,
  and all 183 tests. The clone was removed after verification.
- The post-proposal CI/security repair at `86e53bf` passed the complete GitHub
  Actions run `34318501572` on 2026-09-09. All stages passed, including the
  496-file full-history/privacy audit, frozen evaluator/source binding, the
  narrow reviewed application-dependency policy, all 227 tests, coverage, and
  the exact Moshi advisory-drift audit.
- The bounded-evidence hardening at `72d531a` passed complete GitHub Actions run
  `34687281188` on 2026-09-12, including 351 tests, full-history/privacy,
  deterministic evidence-summary binding, dependency audits, and the pinned
  scientific-risk drift gate.
- The consented-session persistence hardening at `a02363a` passed complete
  GitHub Actions run `34688277409` on 2026-09-12, including 358 tests and every
  compile, lint, type, privacy/history, frozen-source, dependency, coverage, and
  scientific-risk gate.

## Accepted pinned-training compatibility risks

The application-policy exception above is separate from the older isolated
Moshi environment risks below. Neither passing audit means “no
vulnerabilities”; each means there is no unreviewed drift from its explicit
policy.

The general application audit has the single reviewed exception above. The
isolated scientific Moshi lock is separate: after upgrading its independently
patchable packaging tool from pip 24.0 to 26.2, its policy records 56 reviewed
advisory identities in three upstream-constrained packages. Raw scanner-row
counts are not stable because vulnerability databases can publish duplicate
records or change which CVE/PYSEC/GHSA alias is primary.

| Package | Pinned reason | Required mitigation |
|---|---|---|
| `torch==2.6.0` | Moshi-Finetune requires exactly 2.6 and Moshi requires <2.7 | Load only pinned safetensors/project checkpoints; no untrusted `torch.load`; local single-GPU research only |
| `aiohttp==3.11.18` | Moshi requires <3.12 while fixes start outside that range | No public service; bind later research servers only to loopback or a trusted authenticated network |
| `sentencepiece==0.2.0` | Moshi requires 0.2 while the fix is 0.2.1 | Parse only the SHA-256-pinned tokenizer model |

`configs/moshi_dependency_risk_policy.json` records every reviewed advisory,
constraint, and mitigation. `scripts/audit_moshi_dependencies.py` compares the
live advisory database with that exact baseline and fails CI on any package,
version, added advisory, removed advisory, or malformed-policy drift. Its report
is `results/security/moshi_dependency_audit.json`. Matching is by the scanner's
complete advisory identity group, so a database-only primary-ID migration does
not create a false drift failure; an unmatched ID or alias group still fails
closed. A passing drift audit means
“no unreviewed change”; it deliberately does not claim “no vulnerabilities.”
The exception must be removed or reviewed again before public deployment or an
upstream Moshi/trainer migration.

## Deliberately unresolved decisions

- The project-owned source and documentation are now licensed under
  Apache-2.0. This does not alter the separate model, voice, source-corpus,
  generated-derivative, participant-material, and adapter rights documented in
  `THIRD_PARTY_NOTICES.md`.
- Protected main-branch rules cannot be enabled for this private repository
  under the current GitHub plan: the branch-protection API returns HTTP 403.
  The remaining options are GitHub Pro (or another eligible plan) or making the
  repository public; neither change is authorized implicitly.
- A logged-out public-clone inspection is impossible while the repository is
  private. CI plus a fresh authenticated/local clone can verify the same tracked
  content, but public visibility remains a separate student decision.
- General application requirements intentionally use compatible lower bounds
  and are re-resolved/audited in CI. The scientific Moshi trainer is the
  stricter reproducibility boundary and remains exactly locked. A platform-wide
  hashed application lock may be added for a chosen deployment OS, but should
  not be misrepresented as portable across CUDA, CPU, and Python variants.

These residual items are visible policy/deployment choices, not silently passed
engineering gates.
