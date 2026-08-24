# Repository security and reproducibility review

Status date: 2026-08-24

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
- CI compiles and lints `src`, `tests`, and `scripts`; mypy checks all 47
  source files against the Python 3.10 target.
- The measured branch-aware coverage is 58%; the enforced floor is 55%, raised
  from 35% while retaining a small non-flaky margin. Core conversation,
  Moshi-export, QA-policy, rights, and preflight modules are substantially above
  the aggregate.
- `pip-audit --requirement requirements.txt --strict` reported no known
  vulnerabilities on 2026-08-24 and is an enforced CI gate.
- Critical Moshi assets, upstream revisions, and the isolated training
  environment are independently pinned and hash-checked by
  `requirements-moshi.lock`, `third_party/UPSTREAMS.lock.json`, and the
  hardware/environment evidence.

## Deliberately unresolved decisions

- No source-code license has been selected. Until the student/supervisor chooses
  one, the absence of a license grants no public reuse permission. This is kept
  distinct from model, voice, source-corpus, and adapter rights documented in
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
