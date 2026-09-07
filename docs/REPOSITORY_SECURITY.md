# Repository security and reproducibility review

Status date: 2026-09-06

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
- GitHub Actions has read-only repository permissions, fetches complete history,
  and pins the first-party checkout and Python setup actions to immutable commit
  SHAs.
- GitHub dependency vulnerability alerts and automated security fixes are
  enabled; the authenticated API reports alerts available and automated fixes
  `enabled=true`, `paused=false`.
- CI compiles and lints `src`, `tests`, and `scripts`; mypy checks all 49
  source files against the Python 3.10 target.
- The measured branch-aware coverage is 65%; the enforced floor is 60%, raised
  from 35% while retaining a small non-flaky margin. Core conversation,
  Moshi-export, QA-policy, rights, and preflight modules are substantially above
  the aggregate.
- `pip-audit --requirement requirements.txt --strict` reported no known
  vulnerabilities on 2026-09-06 and is an enforced CI gate.
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

## Accepted pinned-training compatibility risks

The general application audit is clean. The isolated scientific Moshi lock is
separate: after upgrading its independently patchable packaging tool from pip
24.0 to 26.2, the audit reports 58 rows (56 unique advisory identifiers) in
three upstream-constrained packages.

| Package | Pinned reason | Required mitigation |
|---|---|---|
| `torch==2.6.0` | Moshi-Finetune requires exactly 2.6 and Moshi requires <2.7 | Load only pinned safetensors/project checkpoints; no untrusted `torch.load`; local single-GPU research only |
| `aiohttp==3.11.18` | Moshi requires <3.12 while fixes start outside that range | No public service; bind later research servers only to loopback or a trusted authenticated network |
| `sentencepiece==0.2.0` | Moshi requires 0.2 while the fix is 0.2.1 | Parse only the SHA-256-pinned tokenizer model |

`configs/moshi_dependency_risk_policy.json` records every reviewed advisory,
constraint, and mitigation. `scripts/audit_moshi_dependencies.py` compares the
live advisory database with that exact baseline and fails CI on any package,
version, added advisory, removed advisory, or malformed-policy drift. Its report
is `results/security/moshi_dependency_audit.json`. A passing drift audit means
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
