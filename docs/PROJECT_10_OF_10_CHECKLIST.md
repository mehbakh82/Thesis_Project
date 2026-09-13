# Checklist for a defensible 10/10 thesis project

Status date: 2026-09-12

This is the authoritative closure checklist. Mark an item complete only when its
named artifact exists and its acceptance test passes. Implemented code,
synthetic fixtures, estimates, and plans are not experimental evidence unless
an item explicitly says they are.

## Closure rule and current baseline

- `[x]` means completed and supported by an artifact or test.
- `[ ]` means incomplete, even if its supporting code is implemented.
- Items owned by a reviewer, supervisor, participant, or hardware provider
  cannot be fabricated or silently waived by code. A documented student waiver
  may narrow the active training scope, but it does not complete the original
  item or authorize the corresponding strict claim.
- A written supervisor-approved scope amendment can replace a requirement; an
  informal assumption cannot.

## Post-release model/data selection hardening

- [x] Bind the newly supplied detailed proposal by SHA-256 and add a
  fail-closed requirements-alignment receipt. All five primary objectives are
  traceable, four have substantial implementation, and zero fully pass their
  strict proposal-level acceptance criteria; see
  `docs/PROPOSAL_ALIGNMENT_AUDIT.md`.

- [x] Replace ambiguous “best model” wording with an executable,
  requirements-constrained audit that separates project-verified evidence,
  official documentation, failures, and unknowns.
- [x] Add current direct candidates: PersonaPlex, Qwen3-Omni, MiniCPM-o 4.5,
  Covo-Audio-Chat-FD, BayLing-Duplex, and DuplexOmni.
- [x] Recheck official direct-model sources on 2026-09-10: record Qwen3-Omni as
  failing Persian speech output because Persian is absent from its published
  output-language list; record MiniCPM-o 4.5 public adaptation as documented
  through LLaMA-Factory/SWIFT while retaining unknown Persian/full-duplex
  adaptation and output. Zero direct candidates satisfy all five gates.
- [x] Add current component challengers: Qwen3-ASR 1.7B/0.6B, Shenava
  Koochik, Rade-ASR-CTC-3B-fa, Omnilingual-ASR, local Qwen3.5-4B/0.8B,
  Qwen3-TTS, and MOSS-TTS-Nano-Persian.
- [x] Preserve the frozen Qwen3-4B result and prohibit reuse of its opened final
  panel for Qwen3.5 selection.
- [x] Add `audit-model-selection` with fail-closed source validation and unit
  tests; generate `results/model_selection_audit.json`.
- [x] Add `audit-conversation-balance` with exact source/session/hour shares,
  remediation arithmetic, and unit tests; generate
  `results/conversation_balance_audit.json`.
- [x] Quantify the final imbalance: Tabaghe16 is 74.9481% of source-pair hours;
  a fixed-size repair replaces 24.694900 h, while a retain-all repair adds
  44.899818 non-Tabaghe hours and remains under 200 h.
- [x] Run NeMo versus Qwen3-ASR-1.7B/0.6B on a newly frozen, leakage-conscious
  40-row Persian development panel (20 Digiato + 20 Zoomit; 272.465 seconds).
  NeMo won decisively: CER/WER 0.230119/0.371593 versus
  0.383788/0.651363 for 1.7B and 0.492509/0.807747 for 0.6B. The frozen rule
  retained NeMo and kept the separate 40-row final panel sealed.
- [x] Run an explicitly post-hoc Shenava-Koochik supplemental screen on the
  already-open ASR development panel without accessing the final panel.
  Shenava completed 40/40 rows and was substantially faster on CPU (median RTF
  0.016045 versus 0.095592), but its CER/WER 0.241644/0.403156 was worse than
  NeMo's 0.230119/0.371593; the paired CER interval
  [-0.016063, 0.044917] crossed zero. NeMo remains selected and no new
  confirmation panel is warranted.
- [x] Attempt the exact Rade-ASR-CTC-3B-fa runtime without misreporting a model
  result. Torch 2.8.0, fairseq2 0.6, Omnilingual-ASR 0.2.0, CUDA, and the
  inference pipeline imported successfully; the deadline-bounded 6.16 GB
  checkpoint transfer did not complete, so inference was not attempted and
  project Persian quality remains unknown. See
  `results/hardware/rade_asr_runtime_attempt.json`.
- [x] Run Qwen3-4B versus Qwen3.5-4B/0.8B on a newly frozen equal-channel,
  session-disjoint 40-row development panel. All three arms completed 40/40
  generations and 80/80 judge calls. Qwen3.5-4B produced zero mean relevance
  gain and only 6/40 row wins; Qwen3.5-0.8B regressed. The frozen rule retained
  Qwen3-4B and kept the separately locked 40-row final panel unopened.
- [x] Build a versioned balanced corpus without overwriting the frozen training
  corpus or mixing result versions. Balanced v2 uses a disjoint 105-episode
  Digiato/Zoomit supplement and deterministic session-fair selection; it passes
  at 6,551 pairs / 110.374 h / four channels / 186 sessions, with Tabaghe16 at
  54.0%, zero missing files, zero reused spans, and zero split leaks. It is
  explicitly not attributed to the already completed Moshi runs.
- [x] Evaluate a second genuinely Persian TTS on identical text and automatic
  latency/intelligibility metrics. On a new equal-channel 40-row panel,
  Mana-Piper achieved CER/WER 0.174401/0.342520 versus
  0.250892/0.494094 for Meta MMS-TTS Persian; both completed 40/40 rows. The
  paired 95% CER-difference interval [0.029315, 0.130514] retained Piper and
  kept the separately locked final panel sealed.
- [ ] Complete human Persian naturalness/pronunciation listening for the TTS
  choice. Automatic ASR round-trip evidence cannot complete this item.
- [ ] Promote a replacement direct model only after public adaptation,
  project-verified Persian output, native duplex runtime, and physical 24 GB fit
  all pass; currently no candidate qualifies.

The active conversation-data policy is the transparent, student-authorized
automatic-only waiver in `docs/QA_WAIVER.md`. It permits limited internal
training but does not make this a strict 10/10 project. Strict 10/10 still
requires the reviews or a formal supervisor-approved scope amendment.

The project is 10/10 only when every formal requirement in the traceability
table passes, every critical-path item below is closed, all numbers reproduce
from the frozen commit and archived evidence, and no synthetic/component/H100
result is presented as official live/4090/human evidence.

Verified now:

- [x] Formal data, model, duplex, latency, detector, hardware, and human-study
  requirements are represented by explicit gates.
- [x] Working NeMo ASR → exact local Qwen → Piper cascade and a
  continuous-microphone browser interruption-control path. The frozen
  proper-user-channel validation passed 9/9 predeclared rows from a
  source-session-group-isolated split with zero fallbacks.
- [x] Final responder promoted to exact Qwen3-4B-Instruct-2507 prompt-v2. Its
  seven-row frozen eligibility stage passed, then its 40-row/five-session
  final test passed all seven automatic gates with 240/240 valid judge calls:
  relevance 3.150 versus 1.425, coherence 3.325, gains +1.725/+1.800,
  paired relevance win rate 67.5%, and relevance ≥2 on 87.5% of rows.
- [x] Legacy 7 MB reconstruction checkpoint is blocked from being described or
  loaded as a genuine direct speech-language model.
- [x] Direct path selected: pinned Moshika 7B with the official
  Moshi-Finetune LoRA trainer.
- [x] Moshika, Mimi, SentencePiece, trainer, runtime, and Persian Piper files
  and revisions are pinned and hash-verified.
- [x] Real one-step H100 wiring smoke passed model/Mimi load, LoRA init,
  tokenization, loss, backward, and optimizer: loss 4.614331, peak 15.258 GB.
- [x] Full inventory: 775.887 caption hours / 1,442 episodes.
- [x] Authorized staging: 309 episodes / 1,129 windows / 219.946 candidate
  hours / 207.154 automatic multi-speaker hours / 242.445 aligned hours.
- [x] Built corpus: 6,754 non-reused pairs / 123.796 source-pair hours.
- [x] Built and independently audited balanced v2: 6,551 pairs / 110.374
  source-pair hours / 186 sessions / four channels; largest source 54.0%.
- [x] Internal thesis training is supervisor-approved for all 1,129 windows;
  raw-data redistribution remains prohibited.
- [x] 770 conservative interaction candidates recovered: 717
  interruption-like and 53 backchannel-like.
- [x] Best-practice QA sheets, Persian guides, stratified samples, and playback
  helper were prepared and are preserved unchanged.
- [x] Student QA waiver is machine-readable, hash-bound, fail-closed, and tested;
  it explicitly disables human-verification, verified-interruption, strict
  coverage, and supervisor-waiver-approval claims.
- [x] Git identity is Mehran Bakhtiari; private planning/definition documents,
  raw data, environments, model blobs, checkpoints, and credentials are
  excluded from tracking.
- [x] The full local audit passes compile, Ruff, mypy, the complete test suite,
  branch-aware coverage with its CI floor, full-history/privacy/secret checks,
  general dependencies, the accepted scientific-risk baseline, and all ten
  upstream pins. The final release audit below records the definitive counts.
- [x] General dependencies have exactly one narrow reviewed exception
  (Accelerate 1.15.0 / conservatively carried CVE-2026-69112 risk, with no
  declared fixed release). The official 1.15.0 wheel retains the relevant
  unvalidated path join. If the scanner feed omits it, CI still requires the
  exact pinned version and preserves it as a known scanner-unmapped risk; all
  other findings and any version/policy drift fail closed. The pinned
  scientific lock has a separate exact accepted-risk baseline and mitigations.
- [ ] Window QA: 0/40 reviewed.
- [ ] Interaction QA: 0/24 reviewed.
- [x] Waiver-bound Moshi export/audit: 6,754 pairs / 108.584 measured stereo
  hours, zero machine failures, 24/24 deterministic assistant re-syntheses.
- [x] Exact full-profile probe: loss 3.808453, 22.707 GB peak, fused AdamW,
  and a 967 MiB/699-tensor CPU-offloaded adapter save; all gates pass.
- [x] Scientific H100 experiments are preserved honestly: v1 passed
  teacher-forced held-out controls but failed official-runtime generation; v2
  completed all 20 candidates, v3/v4 completed all five candidates, and v5
  completed all five reduced-semantic-codebook-weight candidates; v2 through
  v5 failed their unchanged autoregressive eligibility gates before final-test
  access. V6/v6.1/v6.2 then completed bounded train-only capacity diagnostics;
  v6.2 passed its 30% learning gate with a 32.20% text-loss reduction but all
  four candidates passed 0/9 direct-runtime rows, so no direct adapter is
  promoted.
- [x] The earlier cascade v4 train panel passed 9/9 but was subsequently found
  to use assistant channel 0 as ASR input. It is retained as a component-chain
  smoke test. Before reading any validation row, the channel was corrected to
  audited user channel 1 in a frozen protocol; the unchanged real-service
  cascade then passed 9/9 group-disjoint validation rows with zero rule
  fallbacks and no final-test access.
- [x] Real-audio automatic-label detector proxy completed once on 132 balanced
  events / 22 held-out sessions: accuracy 81.06%, interrupt F1 78.99%, FAR
  9.09%, and FRR 28.79%; it is explicitly ineligible as independent ground
  truth because human-verified labels remain zero.
- [ ] Physical-4090 evidence, independently labeled detector evidence,
  perceptual model review, and human study are pending.
- [x] GitHub CLI authentication is persistent for `mehbakh82`; the remote is
  configured without placing credentials in the repository.
- [x] Apache-2.0 is selected for repository-owned source and documentation;
  third-party, model, voice, data, adapter, restricted-media, and participant
  terms remain explicitly separate.
- [x] Local Qwen3.5-0.8B and Qwen3.5-4B checkpoints passed bounded synthetic
  Persian text-only compatibility smokes. Neither is promoted or scored on the
  opened final panel; they remain future low-memory/upgrade candidates.

## What remains now

The project now has the positive automatic semantic result needed for
submission. The frozen Qwen3-4B prompt-v2 cascade—not direct Moshi—is the
production candidate. Direct Moshika remains a valuable negative ablation.

The final full and concise manuscripts are now tracked under `thesis-report/`,
reconciled to the frozen evidence, and independently rebuilt. No further
training or engineering-owned manuscript task is required for the current
submission. The remaining student/external actions are:

1. Submit the required PDF edition and the repository/tag identified in the
   release handoff.
2. If and only if a physical 12–24 GB target GPU becomes available before a
   later evidence freeze, run the already documented live-browser
   hardware/latency protocol.

The strict-rubric research gaps remain independently human-reviewed detector
labels, the 5–10-person study with two participants aged 60+, the waived
listening reviews, official physical-target latency, and a deployment-eligible
direct Moshi adapter if the rubric requires the direct architecture
specifically. These cannot be fabricated. They must be reported as limitations
if unavailable at submission.

The 40-row window review, 24-row interaction review, and 24-pair synthesized-
assistant listening review are explicitly **waived, not completed** for the
active limited-scope run. Their tools, sheets, and best-practice instructions
remain preserved. This is not an active coding task, but it permanently blocks
claims that the corpus or interruptions were human-verified and means the
original unwaived rubric is not fully satisfied. Branch protection is also
unresolved because GitHub returns HTTP 403 for this private repository under
the current plan; it is a platform/owner decision, not a scientific evidence
gate.

## Critical path

The original strict path remains:

```text
40-row + 24-row review
        -> apply QA and audit final response pairs
        -> export immutable Moshi stereo data
        -> exact-shape one-step H100 memory probe
        -> full H100 LoRA training
        -> held-out model validation and direct-runtime integration
        -> physical-4090 live evaluation and human study
        -> final analysis, evidence freeze, thesis, and release
```

The active limited-scope path is:

```text
documented student QA waiver + preserved QA assets
        -> automatic-only pair build/audit with false strict claims
        -> waiver-bound Moshi export and exact-shape probe
        -> limited internal training/evaluation with explicit limitation
        -> 4090 evidence, remaining thesis evidence, and release
```

## 1. Freeze supervisor decisions before viewing final outcomes

Owner: student and supervisor. Record dated answers in
`docs/SUPERVISOR_DECISIONS.md`; keep original correspondence privately.

- [x] Approve private, non-commercial training on the crawled YouTube corpus.
- [ ] Decide which 100–200-hour measure is binding. Conservative recommendation:
  require the final exported conversational training duration—not raw episode,
  staging, candidate, or estimated duration—to be in range.
- [ ] Approve the 40-row stratified window QA sample, or specify the replacement
  sample size and acceptance threshold.
- [ ] Approve the 24-row / 168.3-second interaction-boundary sample, its minimum
  precision, and the minimum verified-interruption count.
- [ ] Confirm that reviewed existing podcast/interview interactions satisfy the
  interruption-data requirement without new student recordings.
- [x] Record the student’s time/no-delegate QA waiver separately from supervisor
  data-use approval; explicitly state that supervisor approval of the waiver is
  not claimed.
- [ ] Confirm pinned Moshika 7B + official Moshi-Finetune LoRA as the accepted
  open-base adaptation path.
- [ ] Decide whether 500 ms binds maximum, p95, or median. Until then, use the
  literal conservative test: every included official turn ≤500 ms, while also
  reporting p50, p95, maximum, failures, and timeouts.
- [ ] Decide event-level versus frame-level detector accuracy. Recommended
  primary result: event-level, speaker/session-held-out accuracy, plus
  interrupt precision/recall/F1, FAR, and FRR.
- [ ] Confirm a physical RTX 4090 (24 GB) as eligible target hardware.
- [ ] Approve consent/ethics procedure and study retention mode: `features`
  recommended; `metrics` if acoustic aggregates are prohibited.
- [ ] If participant recruitment is impossible, obtain a written scope
  amendment before substituting expert/listening evaluation. Otherwise the
  human-study requirement remains incomplete.
- [ ] Decide how to submit the “dataset” without redistribution rights.
  Recommended: private audited corpus plus public schemas, construction code,
  hashes, aggregate statistics, and a restricted-data statement.
- [ ] Confirm final deliverables: repository, private data handoff if any,
  adapter, evidence bundle, thesis PDF, and demo.

Exit: every applicable ambiguity has a dated answer recorded before outcome
selection.

## 2. Publish and protect the repository

Owner: student for authentication; Codex can push afterward.

- [x] Remote is `https://github.com/mehbakh82/Thesis_Project.git`.
- [x] Existing commits use
  `Mehran Bakhtiari <94431009+mehbakh82@users.noreply.github.com>`.
- [x] `cursor_bsc_thesis_project_planning.md` and `تعریف پروژه.docx` are
  root-ignored and absent from tracked history.
- [x] Raw media, internal manifests, environments, caches, credentials,
  checkpoints, and large weights are ignored.
- [x] GitHub CLI authentication is persistent for `mehbakh82` without
  placing credentials in the repository.
- [x] Push `main`; require local `main` and `origin/main` to resolve to the
  same commit at every handoff.
- [x] Verify a fresh authenticated clone of the private remote: clean worktree,
  expected HEAD, one author/committer identity, and passing full-history
  artifact/privacy/secret audit.
- [x] GitHub Actions uses read-only permissions and immutable first-party Action
  SHAs; it compiles, lints, type-checks, audits, tests, and enforces 75%
  branch-aware coverage.
- [x] CI lints `src`/`tests`/`scripts`, parses tracked JSON/YAML/TOML,
  rejects forbidden/oversized artifacts and private paths, scans all reachable
  revisions for secrets, audits general dependencies, and fails on scientific
  dependency-risk drift.
- [x] Remove private absolute workstation paths from the current candidate
  evaluators without rebinding frozen outcomes. The portable files and exact
  historical blobs/receipt bindings are independently hash-verified in CI by
  `scripts/verify_frozen_evaluator_sources.py`.
- [x] GitHub dependency vulnerability alerts and automated security fixes are
  enabled and verified through the authenticated API.
- [x] Review Accelerate CVE-2026-69112 and the later 1.15.0 wheel after no fixed
  release became available. The 1.15.0 code retains the relevant path join, so
  CI pins that exact wheel-era version under trusted hash-inventoried local-model
  controls and fails closed if scanner-feed omission is accompanied by missing
  or changed resolved-package evidence.
- [ ] Enable protected/CI-gated main-branch updates. Revalidated on 2026-09-09:
  GitHub returns HTTP 403 for both classic branch protection and repository
  rulesets on this private repository under the active plan; changing plan or
  visibility requires the student.
- [x] Confirm Apache-2.0 as the source-code/documentation license;
  third-party/model/data/adapter terms remain separate in
  `THIRD_PARTY_NOTICES.md`.

Exit: remote equals the frozen local commit, CI is green, and a fresh clone at
the approved visibility is privacy-safe.

## 3. Complete both human listening reviews

Owner: student, supervisor, or approved reviewer with archive access. No new
speech recording is required.

Window QA:

Status for the current limited run: deliberately waived, **not completed**.
All items and artifacts below remain available for later strict recovery. See
`docs/QA_WAIVER.md`.

- [ ] Read `docs/MANUAL_QA_FA.md`.
- [ ] Review all rows in
  `data/processed/manifests/conversation_manual_qa.csv`.
- [ ] Set every `review_status` to `pass` or `fail`; provide `reviewer_id`.
- [ ] For every pass, complete speaker count, speaker assignment, caption
  acceptability, and overlap-annotation checks.
- [ ] Note failures/borderline cases; do not alter IDs, paths, automatic fields,
  or provenance columns.
- [ ] Confirm 40/40 complete and zero duplicate IDs.

Interaction QA:

- [ ] Read `docs/INTERRUPTION_QA_FA.md`.
- [ ] Review all rows in
  `data/processed/manifests/conversation_interruption_qa.csv`.
- [ ] Play only the short excerpt with
  `.venv/bin/python scripts/review_interaction_candidate.py --row N`.
- [ ] Check distinct speakers, both turn boundaries, and audible overlap; set
  `corrected_label` to `interrupt` or `backchannel`; provide `reviewer_id`.
- [ ] Reject uncertainty rather than promoting an automatic label.
- [ ] Confirm 24/24 complete, four-channel coverage, and at least one verified
  interruption. If precision fails the agreed threshold, sample a deterministic
  supplement or narrow the claim.

Exit: both internal, gitignored sheets retain original IDs and pass the
fail-closed preflight parser.

## 4. Apply QA and construct the final corpus

Owner: Codex after section 3.

Active waiver alternative:

- [x] Preserve both blank QA sheets, both reviewer guides, and the playback
  helper; do not synthesize reviewer IDs or decisions.
- [x] Validate `configs/conversation_qa_waiver.yaml` and bind its hash to every
  output pair.
- [x] Build 6,754 pairs / 123.796 hours from the authorized automatic manifest
  with `--qa-waiver configs/conversation_qa_waiver.yaml`.
- [x] Run `audit-conversations` with the same waiver and file checks enabled:
  162 sessions, 407 speakers, zero missing files, zero split leaks, and zero
  reused spans.
- [x] Require `training_ready_under_qa_waiver=true`, while requiring
  `thesis_coverage_ok=false` and all three human/strict claims false.
- [x] Freeze waiver, conversation-manifest, split-manifest, WAV/metadata,
  assistant-voice, export-report, and independent-audit hashes used by training.
- [x] Preserve the completed training lineage while publishing a separate
  recommended balanced-v2 lineage. Its 6,551 pairs / 110.374 source-pair hours
  pass authorization, common-waiver, file, non-reuse, split, and source-balance
  gates; no completed model is claimed to have used v2.

The following is the preserved strict path and remains incomplete:

- [ ] Apply window review:

  ```bash
  .venv/bin/python -m thesis_s2s.cli apply-conversation-qa \
    --in-jsonl data/processed/manifests/conversation_episode_windows_noise_labeled_combined_authorized.jsonl \
    --out-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl
  ```

- [ ] Require `results/manual_qa_report.json`: 40 decisions, zero incomplete,
  zero unknown IDs, and fail-closed behavior.
- [ ] Apply interaction review:

  ```bash
  .venv/bin/python -m thesis_s2s.cli apply-interruption-qa \
    --in-jsonl data/processed/manifests/conversation_episode_windows_reviewed.jsonl \
    --out-jsonl data/processed/manifests/conversation_episode_windows_interactions_reviewed.jsonl
  ```

- [ ] Require `results/interaction_qa_report.json`: `qa_complete=true`, zero
  incomplete/unknown IDs, and `verified_interruption_present=true`.
- [ ] Re-audit reviewed windows, preserving channel, noise, alignment, speaker,
  overlap, authorization, and split counts.
- [ ] Build non-reused adjacent response pairs; run `audit-conversations` with
  file checks enabled.
- [ ] Require all final audit gates: 100–200 hours, response pairs, natural
  multi-speaker sessions, ≥2 session groups, group-clean splits, no reused
  intervals, verified interruption, overlap/noise, authorization, manual sample,
  and referenced files present.
- [ ] If final duration falls below 100 hours, add only enough disjoint reserve
  material to restore margin and repeat processing, authorization, QA, and audit.
- [ ] Add Iman Khoraminezhad/Karnakon only if yield or diversity is deficient;
  deduplicate episodes first and never add hours merely for scale.
- [ ] Freeze input/output content hashes when all gates pass.

Exit: `results/conversation_audit.json` has `thesis_coverage_ok=true`; final,
not estimated, duration satisfies the supervisor-approved 100–200-hour measure.

## 5. Export immutable Moshi training data

Owner: Codex after section 4.

- [x] Export the primary Piper set under the documented QA waiver:

  ```bash
  .venv/bin/python -m thesis_s2s.cli export-moshi-data \
    --assistant-audio-mode piper \
    --qa-waiver configs/conversation_qa_waiver.yaml
  ```

- [x] Require group-isolated train/validation/test manifests (6,419 / 131 /
  204), all 6,754 selected pairs exported, **108.584 measured stereo hours**,
  complete authorization, pinned assistant voice, immutable hashes, and raw
  redistribution disabled.
- [x] Independently verify every WAV/header/metadata/hash, source-user PCM and
  offset/silence placement, split isolation, transcript association, and actual
  channel order; `results/moshi_export_audit.json` has zero failures.
- [x] Re-synthesize a deterministic, stratified 24-pair assistant sample with the
  pinned voice: 24/24 exact PCM matches. Preserve the generated unreviewed
  listening sheet.
- [ ] Listen to that stratified sample across channel/split/noise/overlap/
  interaction/response-length conditions. This remains waived—not completed.
- [x] Confirm by content, not metadata alone, that channel 0 is assistant and
  channel 1 is user.
- [x] Keep `--assistant-audio-mode source` as a named multi-voice ablation only.
- [x] Require both export and independent audit
  `training_ready_under_qa_waiver=true`; every waiver-compatible requirement
  passes and all human-verification claims remain false.
- [ ] Require strict `final_training_ready=true`. It is deliberately false
  because `manual_verification_sample_present=false` under the waiver.
- [x] Re-run `gpu-preflight`: `training_data_ready=true`,
  `adaptation_run_ready=true`, and the measured profile plus 4 GiB launch
  headroom gate passed before the full service launch.

Exit: immutable manifests/audio/metadata, assistant-voice hash, statistics, and
passing export report.

## 6. Certify and train the exact H100 profile

Owner: Codex after section 5, without disrupting other GPU users.

Exact-shape launch probe:

- [x] Wait for safe H100 availability; never stop unrelated processes.
  The 2026-08-26 08:09 UTC preflight found 27,934 MiB (27.279 GiB) free
  against a measured 22.707 GB peak and 26.707 GB launch requirement. No
  unrelated process was stopped; both preceding optimizer-temporary OOMs are
  preserved as failed, non-scientific attempts.
- [x] Revalidate the isolated environment, both pinned checkouts, model/Mimi/
  tokenizer schemas, configs, and hashes; all 81 exact lock pins match the
  installed Python 3.12 H100 environment.
- [x] Run the exact 20-second, batch-1, four-microbatch, rank-64,
  gradient-checkpointed, embedding-tuning one-step probe:

  ```bash
  MOSHI_DISTRIBUTED_BACKEND=gloo \
    .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
    scripts/moshi_train_entry.py configs/moshi_h100_profile_probe.yaml
  .venv/bin/python scripts/record_moshi_profile_probe.py
  .venv/bin/python -m thesis_s2s.cli gpu-preflight \
    --out results/hardware/current_preflight.json
  ```

- [x] Require finite loss, current hashes, `full_profile_gate_passes=true`, and
  current free VRAM ≥ measured peak + 4 GB.
- [x] Keep this probe labelled `scientific_evidence=false`.
- [x] Exercise the real checkpoint path. The single-GPU launcher copies the
  967 MiB/699-tensor BF16 adapter directly to CPU; the checkpoint-enabled exact
  probe completed without increasing the 22.707 GB logged training peak.


Full run:

- [x] Run in a persistent systemd service because the pinned trainer lacks
  exact optimizer/scheduler/data-loader resume state.
- [x] Use only `configs/moshi_h100.yaml` and immutable waiver-bound exports:

  ```bash
  MOSHI_DISTRIBUTED_BACKEND=gloo \
    .venv-moshi/bin/torchrun --standalone --nproc-per-node 1 \
    scripts/moshi_train_entry.py configs/moshi_h100.yaml
  ```

- [x] Preserve logs, resolved args, environment, input hashes, base hashes, GPU,
  wall time, seed, losses, and periodic adapters.
- [x] Archive failed runs and restart honestly from immutable inputs; never call
  a restart an exact resume.
- [x] Check train/validation loss for finiteness, convergence, instability, and
  overfitting. All 800 logged train losses and all 16 corrected complete-scope
  validation losses are finite and improve through step 8,000. The pinned raw
  evaluator's iterator-exhaustion `NaN`s are diagnosed and excluded.
- [x] Predeclare checkpoint selection in `docs/MOSHI_SELECTION_PROTOCOL.md`:
  minimum finite mean validation `eval_loss` across all expected 500-step
  checkpoints, tie broken toward the earlier step; never use held-out outcomes.
- [x] Implement a fail-closed selected-adapter validator for hashes, saved
  config, exact trainable keys/shapes/dtypes, finite values, and the pinned
  official loader; all execution gates pass.
- [x] Load and hash the selected step-8,000 adapter/config in the pinned runtime:
  699 exact BF16 tensors, 506,753,024 parameters, 337 LoRA layers, no missing,
  unexpected, meta, shape, dtype, or non-finite values.

The first launch on commit `1603beb` was intentionally stopped at step 150
before its first checkpoint after inspection found that the pinned one-GPU
saver would clone the 967 MiB adapter on a GPU with about 1.1 GiB free. Its
logs and metrics are preserved and hash-attested; it is not selection evidence.


V1 autoregressive deployment verdict:

- [x] Launch the selected adapter through the pinned official websocket server
  and attested local client on loopback; all hashes, CUDA load, reconnect,
  static-bundle, Opus, GPU-allocation, and no-fallback checks pass.
- [x] Run the matched unadapted base control. It produces normal English audio
  (RMS 0.057698) and text, proving that the streaming protocol is healthy.
- [x] Reject v1 for deployment: step 500 produces English (zero Persian
  letters), while steps 1,000, 2,000, 4,000, and selected 8,000 produce
  near-silence/no text. Teacher-forced loss is not treated as generation quality.

Corrective v2 run:

- [x] Freeze a new untouched group-held-out final test before v2 training:
  738 rows / 14 sessions / 11.894073 hours; training, validation, and final-test
  sessions are pairwise disjoint and the old v1 test is not reused.
- [x] Freeze `docs/MOSHI_V2_SELECTION_PROTOCOL.md` with all 20 candidates,
  a nine-row validation-only official-server panel, and speech/text/Persian
  eligibility gates before launch.
- [x] Restore the documented 100-second context, increasing response-onset
  coverage from 29.4% to 81.1% of train chunks.
- [x] Complete the exact one-step optimizer/checkpoint probe at 22.797 GB peak.
- [x] Preserve and disclose the first v2 attempt: it trained through step 1,800
  with finite metrics, then failed only while creating checkpoint 1,800 because
  the filesystem returned `ENOSPC`. After the complete clean v2 retry
  superseded it, its 17 checkpoint tensors were intentionally discarded to
  recover 17.231 GB; resolved args, metrics, TensorBoard logs, and hashes remain
  under `checkpoints/moshi_fa_100s_v2_failed_enospc_20260827`, with the complete
  deletion receipt in `results/hardware/storage_cleanup_20260829.json`.
- [x] Launch a clean step-zero retry with the unchanged frozen seed/profile as
  the isolated, journal-attested
  `thesis-moshi-h100-v2-retry1-119f9e8.service`; no optimizer-state-free
  pseudo-resume and no unrelated GPU process termination.
- [x] Implement the hash-bound post-training chain: identical complete-scope
  loss reevaluation, exact adapter schema/finite checks, all-candidate
  official-server validation panel, eligible-only selection, official CUDA
  load, one-time complete final-test controls, deterministic final-test runtime
  diagnostics, and launch/checkpoint attestation.

- [x] Complete the clean 2,000-step v2 retry from
  `configs/moshi_h100_v2.yaml`: step 2,000 and its checkpoint completed at
  2026-08-29 07:28:31 UTC with finite metrics and a 27.7 GB logged peak.
- [x] Reevaluate all 20 v2 candidates on the identical complete 202-chunk
  validation scope. Total loss improves monotonically from 2.312407 at step 100
  to 1.734574 at step 2,000.
- [x] Preserve the first panel and failed selector as invalid eligibility
  evidence: eight of nine five-second inputs ended before their source user
  turns, so silence could be correct turn-taking. The pipeline stopped with
  `test_access_started=false`.
- [x] Freeze `docs/MOSHI_V2_PROTOCOL_CORRECTION.md` before corrected generation
  or final-test access. It deterministically selects nine complete five-second
  prompts using input PCM only and leaves all thresholds, candidates, the 9/9
  requirement, loss rule, and tie break unchanged.
- [x] Apply the corrected complete-prompt official-runtime panel to all 20
  candidates. No checkpoint passed all nine unchanged gates; step 400 was best
  at 2/9, steps 500–800 and 1,000 reached 1/9, and all others reached 0/9.
- [x] Finalize v2 selection fail-closed with zero eligible checkpoints and a
  null selection. The infrastructure-only root/NumPy failure is preserved
  separately from the successful runtime report and repaired reproducibly.
- [x] Stop before selected-adapter validation or any final-test evaluation, as
  required by the frozen protocol. The fresh 738-row / 14-session final test
  remains untouched.
- [x] Freeze the bounded v3 embedding-preserving protocol before training:
  restore the pinned upstream rank-128 / `ft_embed=false` LoRA choice; keep
  data, context, optimizer, seed, and all output gates unchanged; screen only
  steps 100–500.
- [x] Implement a fail-closed v3 preflight that binds v2 failure evidence,
  upstream example/config hashes, exact probe/full shapes, untouched-test
  state, H100 headroom, and a conservative disk budget.
- [x] Pass the exact-shape one-step v3 probe, complete all 500 training steps,
  and run all five candidates through identical 202-chunk complete-scope loss
  and unchanged nine-row complete-prompt runtime gates. Loss improves from
  2.084004 to 1.879258; runtime pass counts are 0/9, 0/9, 0/9, 1/9, and 0/9.
- [x] Apply the conditional final-test firewall: v3 produces zero eligible
  checkpoints, selection is null, and selected-adapter/final-test evaluation is
  correctly not run. `test_access_started=false` in the hash-bound pipeline and
  training attestations.
- [x] Diagnose the controlled v2/v3 contrast before any new optimizer step:
  v2's upstream `ft_embed=true` saved 25 embeddings (two text and 23 audio),
  whereas v3 saved only rank-128 LoRA tensors. Freeze the falsifiable v4 change
  as rank-128 LoRA plus exactly `text_emb.weight` and
  `depformer_text_emb.weight`, with all audio embeddings frozen.
- [x] Implement the hash-bound v4 protocol, machine-readable embedding policy,
  exact-schema preflight/probe recorder, validation-only candidate evaluator,
  schema-6 selector, training certificate, and conditional final-test firewall.
- [x] Pass the one-step v4 exact-shape probe at 24.057 GB peak with a
  676-tensor adapter, then complete all 500 bounded steps in 39m53s with five
  exact checkpoints and finite losses.
- [x] Apply the unchanged complete-scope and nine-prompt gates to every v4
  candidate. Fixed-scope loss improves from 2.082125 to 1.879061, but runtime
  pass counts are only 1/9, 0/9, 0/9, 1/9, and 0/9.
- [x] Finalize v4 as a negative result: eligible count zero, selection null,
  certificate passed, and no selected-adapter or final-test stage created.
  `test_access_started=false` remains hash-bound.
- [x] Freeze v5 before training as a one-factor test of v4: reduce only
  `first_codebook_weight_multiplier` from 100 to 10, retaining rank-128 LoRA,
  the exact two text embeddings, frozen audio embeddings, data/split, seed,
  optimizer, horizon, candidate steps, panel, and all thresholds.
- [x] Pass the v5 fail-closed preflight and one-step exact-schema probe, then
  complete 500/500 H100 steps in 42m03s with finite losses and five exact
  676-tensor checkpoints at a 28.640 GB maximum logged peak.
- [x] Evaluate all five v5 checkpoints on the identical 202-chunk validation
  scope and official nine-row runtime panel. Loss improves from 2.082676 to
  1.867859, but pass counts are only 0/9, 0/9, 0/9, 1/9, and 1/9; speech rows
  decline from 9/9 to 4/9 and nonempty outputs remain predominantly English.
- [x] Finalize v5 as a negative result: eligible count zero, selection null,
  training/pipeline certificates passed, and no selected-adapter or final-test
  stage created. `test_access_started=false` remains hash-bound.

Bounded v6 capacity diagnostics:

- [x] Freeze v6 as a 32-row train-only diagnostic explicitly ineligible for
  validation/generalization claims. Expand the Persian text adaptation scope,
  reduce audio-loss weight, preserve all hashes, and keep the final test sealed.
- [x] Complete v6 and its deterministic-text v6.1 follow-up. Preserve their
  negative runtime reports rather than selecting from the diagnostic panel.
- [x] Freeze v6.2 as a one-factor scheduled text-input-dropout test
  (0.25→0.75), with the same base, 32 rows, rank-64 LoRA, full Persian text
  embeddings/output head, audio weight 0.1, four milestones, and unchanged
  nine-row runtime gates.
- [x] Pass the exact v6.2 preflight/probe and complete 200/200 steps with all
  four exact 677-tensor checkpoints, finite metrics, audited FP32 CPU-offloaded
  AdamW, audited dropout at every milestone, and 16.577 GiB peak training
  allocation.
- [x] Re-evaluate all 32 train-only rows identically. Text loss changes from
  0.812474 (step 50) to 0.638987, 0.550852, and 0.551183; the best later value
  is a 32.20% reduction and passes the frozen ≥30% capacity gate. Total loss
  reaches 2.060297 at step 200.
- [x] Exercise all four checkpoints in the exact official server. Every server
  is real and all static adapter checks pass, but every candidate passes 0/9
  output rows; only 1/9 rows per candidate emits both text and speech.
- [x] Finalize v6.2 honestly: positive in-sample learning/capacity evidence,
  negative direct-generation evidence, selection null, no generalization or
  deployment claim, and no final-test access.

The direct-model exit remains open until a future predeclared representative
experiment yields a reproducible adapter that passes scientific validation and
frozen autoregressive Persian-output gates. V1 through v6.2 are finalized and
must not be promoted. This does not reopen the project-level working-prototype
gate: the properly channel-corrected cascade validation closes that gate.

## 7. Validate learning and model quality

Owner: Codex for automation; approved listeners for perceptual checks.

- [x] Run the one-time full 331-chunk group-disjoint test evaluator for paired
  selected-adapter, deterministic LoRA-perturbed, and pinned-base
  text/audio/total losses.
- [x] Prove the adapter path is used with identical-scope loss controls:
  selected total loss 1.727204, pinned base 3.566973, and sign-flipped LoRA
  9.987136; all paired differences are nonzero.
- [x] Prove target sensitivity: controlled masked text/audio target changes
  alter the corresponding fixed-logit losses.
- [x] Audit episode/session/time-span leakage; the one-time test manifest is
  group-isolated, hash-current, absent from training/validation, and was not
  used for checkpoint selection.
- [x] Freeze a separate cascade validation before reading its rows, bind the
  131-row manifest and independent channel/split audit, and predeclare nine
  floor-spaced indices with unchanged v4 component/output thresholds.
- [x] Correct the v4 development evaluator's assistant-channel input before
  validation: the audited export is assistant 0 / user 1, so the validation
  sends channel 1 to ASR. Preserve v4 as a component-chain smoke result.
- [x] Pass the frozen cascade validation 9/9: source hashes, real NeMo ASR,
  exact initialized Qwen, no fallback, ≥80% Persian script, and finite
  non-silent normalized Piper audio all pass on every row; group split leaks
  and final-test accesses are zero.
- [x] Evaluate Persian response relevance/coherence with a documented,
  predeclared rubric: prompt-v2 passed eligibility and then all seven automatic
  gates on 40 final-test rows with 240/240 valid blinded calls. The judge has
  different weights but the same Qwen family and is not a human/independent
  benchmark; WER was not used as the relevance metric.
- [x] Measure synthesized-speech intelligibility separately under a protocol
  pushed before execution: the exact nine cascade outputs reproduced their
  canonical hashes, round-trip NeMo ASR had 0/9 failures, micro CER was 7.14%
  (47/658 edits), and micro WER was 30.41% (52/171 edits). This is explicitly an
  automatic single-voice proxy, not human pronunciation/naturalness evidence.
- [x] Automatically check Persian script, silence-only output, English drift,
  codec collapse/replacement characters, repeated text absence, and unusable
  direct outputs across the frozen panels.
- [ ] Complete human Persian pronunciation, semantic relevance, and naturalness
  review; automatic script/audio gates cannot replace listening.
- [x] Detect and preserve the v1 silence/English-drift failure with matched
  official-server base and checkpoint controls; the negative result triggers
  v2 and is not relabelled as success.
- [ ] Cover clean/noisy/overlap/interruption, long/short turn, out-of-domain, and
  elderly-speech conditions; retain representative failures.
- [x] Compare cascade, unadapted Moshika, and adapted Moshika under explicitly
  separated, hash-bound automatic protocols. Do not imply that the cascade
  mechanics panel and direct-model generation panel measure identical quality.
- [ ] Report the supported source-response multi-voice ablation if useful; add
  hyperparameter ablations only when answering a thesis question.
- [x] Report parameter/trainable counts, H100 time/memory, checkpoint size, and
  the raw-evaluator negative result. Streaming inference real-time factor remains
  part of the physical-4090 evaluation.

Exit: held-out tables, reviewed output sample, error analysis, and proof that the
final adapter causes the learned Persian response behavior.

## 8. Complete direct-model full-duplex integration

Owner: Codex; secure client build complete, live integration after a valid adapter.

Submission decision: the direct branch remains experimental. The cascade is the
working production candidate; browser full-duplex control is implemented, but
the live human/browser acknowledgements below remain uncollected.

- [x] Build the exact pinned Moshi web-client source with the tracked,
  hash-verified security lock overlay in a digest-pinned Node 20 container;
  production audit is clean and all 33 static files/tree hash are attested in
  `results/hardware/moshi_client_build.json`.
- [x] Serve the attested local static bundle with the v1 selected adapter on
  loopback—never a moving remote client. This is a diagnostic smoke, not final
  direct-model readiness.
- [x] Load pinned base/Mimi/tokenizer plus the exact v1 adapter/config in the
  official streaming server and log all hashes. Loading passes; generated
  Persian output fails, so v1 remains disabled.
- [ ] Verify the microphone stays active during assistant playback.
- [ ] Verify interruption stops every browser audio source and produces
  `playback_stopped_ack`.
- [ ] Verify backchannels such as «آها»/«بله» follow the non-stopping policy.
- [x] At the WebSocket transport level, preserve the detector's rolling
  microphone pre-roll, continue collection after automatic/manual interruption,
  and feed the complete captured interruption into the next user turn. The
  regression proves a 3,200-sample pre-roll plus 1,600 continued samples become
  one 4,800-sample next-turn input; physical-browser confirmation remains
  separate.
- [x] Test transport reconnect, explicit cancellation, stale-buffer isolation,
  simultaneous new turns, silence, malformed PCM/JSON/events, and recovery
  after a simulated generation OOM without replacing the live-browser gate.
- [x] Keep model identity fail-closed: direct sessions never silently fall back
  to the cascade, and the submitted cascade exposes any Qwen rule fallback or
  Piper/formant fallback in telemetry. The passing validation used none.
- [x] Add WebSocket end-to-end regression tests for cascade identity/fallback
  telemetry, client acknowledgement ingestion, interruption continuity,
  cancellation, reconnect, state isolation, recoverable errors, and
  metrics-only no-WAV retention.
- [ ] Reproduce direct-adapter identity/load and actual browser source-stop
  acknowledgement end to end. This remains impossible without a
  deployment-eligible direct adapter and physical-browser run.

Exit: repeatable live Persian direct S2S with generation, duplex listening,
interruption, cancellation, and continued conversation using the final adapter.

## 9. Produce real detector evidence

Owner: participants/student for data; Codex for training/analysis.

Completed automatic-label recorded-audio proxy (useful engineering evidence,
not official ground truth):

- [x] Freeze the event, feature, session-split, threshold-selection, and
  one-time-test protocol before feature extraction or model fitting.
- [x] Extract acoustic-only features from authorized real YouTube audio without
  copying or retaining new raw audio; keep session groups disjoint.
- [x] Balance and evaluate 1,048 events: train 710 / validation 206 / held-out
  test 132, with 103 / 24 / 22 disjoint sessions respectively.
- [x] Tune the GBDT threshold on validation only and evaluate the held-out test
  once. At threshold 0.70: accuracy 81.06%, interrupt F1 78.99%, FAR 9.09%,
  FRR 28.79%, confusion matrix `[[60, 6], [19, 47]]`.
- [x] Report event and session-block bootstrap intervals. Accuracy intervals are
  73.53–86.83% and 74.44–87.18%, so neither proves the population accuracy is
  strictly above 80%.
- [x] Compare the fixed energy/ZCR baseline on the same test: accuracy 78.03%,
  interrupt F1 71.84%. Preserve the trained proxy-model hash locally.
- [x] Keep `official_detector_eligible=false`,
  `official_target_satisfied=false`, and `human_verified_labels=0`.
- [x] Enforce the literal strict boundary in code: exactly 80.00% accuracy
  fails `target_ok`, aggregate evidence eligibility, and study readiness; only
  a finite probability greater than 80% passes. The shared predicate also
  rejects missing, malformed, Boolean, non-finite, and out-of-range values;
  dedicated regressions cover both sides of the boundary and invalid inputs.

Official gate still required:

- [ ] Collect consented real events with `features` retention when permitted (no
  WAV). Use `metrics` only when aggregates are prohibited and disclose that it
  prevents feature-based retraining.
- [ ] Include interruption, backchannel, normal turn, silence, cough/non-speech,
  noise, loudness, position, and speaker variation.
- [ ] Keep participant/session groups intact across train/validation/test.
- [ ] Freeze label and threshold protocol before viewing held-out results.
- [ ] Train on training groups, tune only on validation groups, evaluate once on
  held-out groups.
- [ ] Report event accuracy, interrupt precision/recall/F1, FAR, FRR, confusion
  matrix, denominators, and 95% confidence intervals.
- [ ] Require accuracy strictly >80% under the agreed unit; synthetic harmonic
  accuracy cannot satisfy this gate.
- [ ] Compare energy VAD and GBDT on identical held-out events.
- [ ] Report acoustic-onset-to-browser-stop p50/p95/max; retain p95 ≤300 ms as
  the project engineering barge-in target.
- [ ] Analyze errors by age/noise/backchannel/loudness/interruption position.

Exit: real speaker/session-held-out report and detector hash satisfying >80%.

## 10. Run official physical-4090 evaluation

Owner: student provides card; Codex can execute/audit.

- [ ] Copy frozen environment/base/adapter evidence; do not retrain or rediarize.
- [ ] Run `gpu-preflight`; require physical device name 4090, 12–24 GB VRAM,
  CUDA/BF16, and `evaluation_hardware_ready=true`.
- [ ] Never use an H100 memory cap as official target evidence.
- [ ] Verify complete base + adapter + Mimi + runtime fits below 24 GB; report
  peak allocated/reserved VRAM, not file size.
- [ ] Document warm-up, concurrency, browser/audio device, driver, CUDA, Torch,
  and power/performance settings.
- [ ] Measure end-of-speech → client `playback_started` as `T_first_audio`.
- [ ] Measure acoustic interrupt onset → client `playback_stopped_ack` as
  `T_barge_in`; server-only timestamps remain diagnostic.
- [ ] Predeclare timeout policy; include failures/timeouts in denominators.
- [ ] Report p50, p95, max, N, failures, and confidence intervals.
- [ ] Until clarified, require maximum first-audio ≤500 ms; if another statistic
  is approved, still report the literal maximum.
- [ ] Separate cold start, warm turn, components, cascade, direct, and official
  end-to-end rows.
- [ ] Use enough turns/sessions to avoid cherry-picked demo evidence.

Exit: 4090 preflight plus immutable `official_e2e` client telemetry bound to
model and commit hashes.

## 11. Complete the human study

Owner: student, ethics/supervisor, and participants. Code cannot replace it.

- [ ] Obtain required approval/consent procedure before recruitment.
- [ ] Recruit 5–10 native Persian speakers, including at least two aged 60+.
- [ ] Use pseudonymous IDs; store no names/contact data in the repository.
- [ ] Run primary ratings on final direct model and eligible target GPU.
- [ ] Follow predefined script: consent, two warm-ups, time request,
  interruption, non-stopping backchannel, safe noisy condition, three-minute
  free conversation, and complete ratings.
- [ ] Require browser consent before persistence.
- [ ] Prefer `--retention features` and verify no WAV; retain raw voice only with
  explicit approval/consent.
- [ ] Collect all four ratings, MOS/naturalness, satisfaction, interruption
  success, false stops, client timing, and qualitative notes.
- [ ] Record elderly pause/loudness/intelligibility/turn-taking/confusion/recovery
  observations without overgeneralizing from two people.
- [ ] Predeclare exclusion rules; require complete ratings/timing for included
  participants and report all exclusions.
- [ ] Counterbalance baseline/final order if claiming preference differences.
- [ ] Run `export-recordings` and `study-summary` after each block.
- [ ] Require `human_study.json`: N=5–10, ≥2 aged 60+, complete ratings/client
  timing, eligible hardware, and real detector evidence.
- [ ] Report distributions, appropriate means/medians, uncertainty, paired
  comparisons where justified, qualitative themes, and small-N limitations.

Exit: consented, pseudonymized study evidence for naturalness, satisfaction,
interaction, and elderly findings.

## 12. Final analysis, thesis, reproducibility, and release

Evidence/statistics:

- [x] Regenerate `results/eval/EVIDENCE_STATUS.json` from current artifacts
  with a tested fail-closed CLI aggregator that never opens frozen test rows.
- [x] Mark only explicitly consented live-browser rows with client playback
  acknowledgements on physical 12–24 GB hardware `official_e2e`; current count
  is correctly zero.
- [x] Generate current dataset/model/detector/latency/study/ablation/error tables
  from machine-readable results in `results/eval/SUMMARY.md`.
- [x] Include denominators, observed failures, uncertainty where estimable,
  explicit null/reasons where it is not, model seeds, session-group split
  policy, and exact metric definitions in the generated status/report.
- [x] Do not select thresholds, checkpoints, trials, or statistics after seeing
  held-out results; the v2–v5 selectors and the cascade validation are
  predeclared and fail closed. V6–v6.2 are explicitly train-only diagnostics
  and never open a final test.
- [x] Reconcile repository README, review, checklist, and generated-result
  numbers through v6.2 and the channel-corrected cascade validation; retain
  older figures only where explicitly labelled historical/provenance evidence.
- [x] Record the chained post-finalization storage policy: 13 representative
  adapters remain locally (v1: 500/1000/2000/4000/8000; v2: 400/2000; v3:
  400/500; v4: 400/500; v5: 400/500), every retained hash matches committed
  evidence, and 41.01 GiB was reclaimed cumulatively. Candidate counts and
  results are historical certificates; exact recreation of removed negative
  intermediate tensors requires retraining.
- [x] Record four current v6.2 checkpoints separately from the v1–v5 cleanup
  receipt; they are hash-bound by the training, reevaluation, and runtime
  reports and postdate that receipt.
- [x] Reconcile the final full and concise thesis manuscripts and submission
  tables to the frozen evidence. Unavailable external evidence is represented
  explicitly as a limitation rather than estimated or backfilled.
- [x] Prepare a copy-ready Persian manuscript handoff covering the abstract,
  methods, exact result tables, discussion, limitations, conclusion, prohibited
  overclaims, and canonical evidence paths in `docs/THESIS_REPORTING_FA.md`;
  reconcile those boundaries into both tracked manuscript editions.

Thesis narrative:

- [x] Distinguish prior ASR fine-tuning from Moshi response training.
- [x] Document final architecture, channel semantics, LoRA/embedding choices,
  timing, detector ownership, and browser control path.
- [x] Explain H100 training versus physical-4090 official evaluation.
- [x] Document selection, diarization, alignment, conditions, QA, authorization,
  non-redistribution, split isolation, and limitations.
- [x] Compare cascade, base Moshika, adapted Moshika, and justified ablations
  without conflating evidence.
- [x] Include privacy-safe real validation mechanics analysis: 9/9 successes,
  0/9 automatic-gate failures, length/timing distributions, risk counts, row
  extremes, and a hash-bound claim boundary. The panel has no mechanics failure
  examples; semantic/perceptual errors were not inspected.
- [x] Include the separately predeclared automatic intelligibility proxy:
  nine exact reproduced outputs, zero ASR failures, micro CER/WER with complete
  denominators and distributions, no post-hoc threshold, no plaintext, and
  explicit same-ASR/single-voice/N=9 limitations.
- [x] Include the prompt-v2 two-stage automatic semantic result, all
  denominators and thresholds, the 0.5B comparison, the weak automatically
  aligned reference limitation, and the same-family/non-human claim boundary.
- [ ] Add semantic, perceptual, and cautiously scoped elderly error analysis
  only if qualifying human evidence is actually collected; do not fabricate it.
- [x] Verify repository citation metadata and third-party boundaries: all ten
  locked upstreams appear by name and exact revision in
  `THIRD_PARTY_NOTICES.md`; the evaluated Qwen revision is Apache-2.0 and
  lock-matched; the optional installed Piper runtime is GPL-3.0-or-later and is
  separated from the Persian voice/model/data terms; the external Persian ASR
  artifact is explicitly not relicensed. Regression tests enforce this map.
- [x] Verify the final thesis manuscript bibliography: all 50 cited keys resolve
  to exactly 50 tracked bibliography entries, with no missing or unused entry;
  isolated full and concise builds report zero undefined citations or
  references.
- [x] Select Apache-2.0 for repository-owned source/documentation without
  relicensing third-party/model/data/adapter artifacts.
- [x] Ensure repository-provided abstract/conclusion/tables/demo wording makes
  no claim beyond evidence; the tracked full and concise manuscripts preserve
  the same explicit automatic/human, H100/target-hardware, and
  component/direct-model boundaries.

Final engineering audit:

- [x] Run (2026-09-01: Ruff passed; mypy passed 49 source files; 140 tests
  passed; 63% branch-aware coverage; all 8 upstream checks passed; final H100
  preflight correctly rejects official target-hardware evaluation; release
  snapshot regenerated after the evidence commit):

  ```bash
  .venv/bin/ruff check src tests scripts
  .venv/bin/mypy src
  .venv/bin/python -m pytest -q
  .venv/bin/python -m thesis_s2s.cli verify-upstreams \
    --checkouts-root third_party/checkouts
  .venv/bin/python -m thesis_s2s.cli gpu-preflight \
    --out results/hardware/final_preflight.json
  .venv/bin/python -m thesis_s2s.cli release-snapshot \
    --out results/release/final_snapshot.json
  ```

- [x] From a fresh full-history GitHub clone at remote HEAD `cd8e15f`, build a
  wheel, install it into an isolated temporary target without duplicating the
  verified dependency environment, run all 140 tests, verify the installed
  package imports from that target, and remove the temporary checkout.
- [x] Run the post-validation local CI-equivalent audit on 2026-09-05: compile,
  Ruff, and mypy pass; all 169 tests pass in 16.10 seconds; branch-aware
  coverage is 64%; 351 tracked artifacts/history/privacy/secrets pass; general
  dependencies have no known vulnerabilities; the exact scientific
  accepted-risk baseline and all eight upstream pins pass.
- [x] Run the post-release transport audit on 2026-09-06 at `f979619`:
  compile, Ruff, and mypy pass; all 173 tests pass in 16.12 seconds; branch
  coverage is 64%; 353 tracked files pass the history/privacy/secret audit;
  general dependencies have no known vulnerabilities; the exact 56-advisory
  scientific exception baseline and all eight upstream pins pass. Four
  WebSocket-level tests cover continuation, acknowledgements, identity
  telemetry, cancellation, reconnect, simultaneous turns, stale-buffer
  isolation, malformed/silent input, simulated OOM recovery, and metrics-only
  no-WAV retention. The machine-readable receipt is
  `results/release/final_audit.json`.
- [x] Run the post-analysis local audit on 2026-09-06: compile, Ruff, and mypy
  pass; all 175 tests pass in 16.22 seconds; branch coverage is 64%; 357
  tracked files pass the history/privacy/secret audit; general dependencies
  have zero known vulnerabilities; the exact 56-advisory scientific exception
  baseline and all eight upstream pins pass. The current H100 is correctly
  rejected as official 12–24 GB evaluation hardware.
- [x] Verify the prior duplex-ready remote release at `da6fed8`: GitHub CI run
  `34013864466` passed every stage, and a fresh full-history private clone was
  clean at exact remote HEAD with the correct identity, 354-file tracked audit,
  Ruff, mypy on 49 source files, and all 173 then-current tests passing.
- [x] Run the post-attribution local audit on 2026-09-06: compile, Ruff, and
  mypy pass; all 178 tests pass in 17.22 seconds; branch coverage is 64%; all
  358 tracked files pass the history/privacy/secret audit; general dependencies
  have zero known vulnerabilities; the exact 56-advisory scientific exception
  baseline passes; and all nine upstream pins are valid, with both present
  local checkouts matching exactly.
- [x] Run the post-intelligibility local audit at `6b4a7e9` on 2026-09-06:
  compile, Ruff, and mypy pass; all 182 tests pass in 16.43 seconds; branch
  coverage is 65%; all 362 tracked files pass the history/privacy/secret audit;
  general dependencies have zero known vulnerabilities; the exact 56-advisory
  scientific exception baseline and all nine upstream pins pass.
- [x] Run the final Qwen3-4B release audit on 2026-09-07: exact 4B runtime
  initialization and first-attempt Persian generation passed without fallback;
  compile/Ruff/mypy passed; all 205 tests passed in 15.80 seconds; branch
  coverage was 65%; all 384 tracked files passed full-history/privacy/secret
  audit; the general dependency environment had zero known vulnerabilities;
  the exact scientific-risk exception had no drift; and all ten upstream pins
  passed. The 40-row final-test certificate remained hash-verified with 240/240
  valid judge calls and all seven automatic gates passed.
- [x] Repair and verify the post-proposal CI privacy/security regression at
  `86e53bf`: GitHub Actions run `34318501572` passed all stages on 2026-09-09,
  including the 496-file full-history/privacy audit, frozen evaluator/source
  binding, the narrow reviewed application-dependency policy, all 227 tests,
  coverage, and the exact Moshi advisory-drift audit.
- [x] Verify the CI-portable pre-license evidence freeze at `fe32e6a`:
  GitHub Actions run `34029248788` passed every stage, including compile,
  Ruff, mypy, the 362-file history/privacy/secret audit, general dependency
  audit, all 182 tests, coverage, and scientific-risk drift audit. A fresh
  full-history clone was clean at the exact remote commit, contained no private
  validation manifest, used only Mehran Bakhtiari as author/committer, and
  independently passed Ruff, mypy on 49 source files, the tracked-artifact
  audit, and all 182 tests. Its temporary 119 MB clone was then removed.
- [x] Verify a fresh full-history clone at release-snapshot commit `c476e5c`:
  clean tree, exact remote HEAD, only Mehran Bakhtiari as author/committer,
  352-file history/privacy/secret audit passed, Ruff passed, mypy passed 49
  source files, and all 169 tests passed. The temporary clone was removed after
  verification.
- [x] Confirm the complete 13-stage GitHub CI run `33475374049` passes at
  pushed commit `cd8e15f`, including compile, lint, type, history/privacy,
  dependency, test, coverage, and pinned scientific-risk gates.
- [x] Diagnose the red remote CI runs at `b15a38e`: two storage receipts exposed
  the local volume mount string and were correctly rejected by the privacy
  audit. The host-specific strings were removed, the audit reproduced locally,
  and the corrected full remote run `33314972249` passed at commit `24ec60f`.
- [ ] Reproduce a deployment-eligible direct-adapter load and frozen held-out
  inference in a separately provisioned clean scientific environment. This is
  blocked honestly because v1–v6.2 produced no deployable adapter; opening a
  v2–v6.2 final test without an eligible selection remains prohibited.
- [x] Confirm by test that `features`/`metrics` retention modes write no
  WAV.
- [x] Run full-history secret/private-path/size scans, general dependency audit,
  and exact scientific-risk drift audit; document the accepted pinned-stack
  risks and mitigations in `docs/REPOSITORY_SECURITY.md`.
- [x] Select and record Apache-2.0 for repository-owned source/documentation;
  do not infer rights for third-party/model/data/adapter artifacts from it.
- [x] Parse all tracked JSON/JSONL/YAML/TOML and enforce their schemas where
  applicable.
- [x] Verify every currently finalized manifest, export, retained adapter,
  evaluation, audit, and release-snapshot hash. The snapshot is generated only
  from a clean parent commit and the fresh remote clone is independently tested.
- [x] Repeat the final snapshot/tag verification after selecting the source
  license; the Apache-2.0 handoff is frozen as
  `submission-final-apache2-2026-09-07`.
- [x] Repeat the local audit and final snapshot/tag process after adding the
  manuscript artifact: the untouched report source tree and both final PDFs
  are tracked under `thesis-report/`; an isolated rebuild produced a clean
  100-page full edition and a clean 29-page concise edition. The PDF hashes,
  page counts, source-tree object, retained CC-BY-4.0 template attribution,
  and Apache-2.0 boundary are recorded in the final audit.
- [x] Confirm no raw YouTube media, private source documents, participant
  identifiers, credentials, environments, caches, or restricted weights are
  tracked in any reachable revision.
- [x] Freeze and push the current pre-license evidence bundle: configs, locks,
  hashes, environments, audits, metrics, logs, approved aggregates, generated
  tables, privacy-safe descriptive analysis, and the automatic intelligibility
  proxy are snapshot-bound; restricted media remains private.
- [x] Tag and push the pre-license submission handoff. The final reconciled
  release-record commit is frozen as `submission-final-prelicense-2026-09-06`;
  its full-history clone, identity, snapshot, tracked-artifact audit, test
  suite, remote parity, and tag-triggered CI are independently verified.
- [x] Freeze the positive Qwen3-4B release as
  `submission-final-qwen4b-2026-09-07`; its final snapshot binds the new
  protocol, eligibility report, final-test report, runtime, attribution,
  aggregate evidence, audit receipt, and thesis-ready documentation.
- [x] Freeze the Apache-2.0 closeout as
  `submission-final-apache2-2026-09-07`; its regenerated evidence, audit,
  snapshot, branch/tag CI, and fresh full-history clone verification preserve
  the positive Qwen3-4B result and the non-promotion boundary for Qwen3.5.
- [x] Preserve the initial manuscript release as
  `submission-final-reports-2026-09-07`, then freeze the validated readability,
  RTL-link, and architecture-diagram refinements as
  `submission-final-reports-v2-2026-09-07`. Exact chat transcript filenames
  and LaTeX auxiliary outputs remain ignored, while both final PDFs, LaTeX
  sources, bibliography, figures, fonts, and template support files are
  release-bound.
- [x] Freeze the post-training source-balanced corpus and reconciled evidence
  handoff at commit `f53e602` under the annotated tag
  `submission-balanced-v2-2026-09-09`. The clean full-history clone passed all
  235 tests, Ruff, mypy over 51 source files, the 510-file history/privacy/
  secret audit, frozen-source receipt verification, and both dependency-risk
  gates. Branch and tag-triggered CI runs `34366895680` and `34367625360`
  passed on that exact commit.
- [x] Remove the only non-failing CI annotation after the evidence freeze. The
  optional pip cache attempted to archive the large PyTorch/CUDA wheel cache
  and exhausted hosted-runner scratch space after all checks had passed;
  cache-only follow-up `6edb522` disables that archival. Warning-free run
  `34368546413` then passed every substantive gate without changing any model,
  corpus, evaluation, or manuscript artifact.
- [x] Remove the unused restricted legacy YAML parser and test the sole safe
  PyYAML loader, portable-path serialization, consent/session identity checks,
  retention enums, the complete study-readiness conjunction, and the strict
  detector threshold. Study aggregation independently revalidates rating
  consent, session/speaker/age linkage, and all four integer 1–5 bounds; any
  invalid row blocks readiness. The aggregate evidence gate also requires all
  eight canonical study requirements instead of trusting a standalone
  `official_ready` flag. The current privacy-safe receipt has one participant,
  zero turns, zero valid/complete ratings, and one invalid row, so it remains
  explicitly ineligible. The full local suite passes 248 tests with 69%
  branch-aware coverage; configuration coverage rises from 26% to 97% and the
  privacy-sensitive session logger from 50% to 90%.
- [x] Replace the aggregate evidence report's historical hard-coded
  `immutable_submission_tag_created: false` with a fail-closed release
  attestation. `results/release/submission_tag_attestation.json` now binds the
  definitive annotated tag `submission-final-audited-v3-2026-09-13`, tag object
  `ab5a175`, target commit `a3f36eb`, and successful branch/tag CI runs
  `34742816867` and `34742972132`.
  The tracked-artifact CI audit independently resolves the tag object and peeled
  commit from Git and requires the attested commit to be an ancestor of `HEAD`.
  Direct temporary-repository regressions cover a valid annotated tag, a
  mismatched object hash, and malformed receipt JSON. The enforced branch-aware
  coverage floor is raised from 60% to 68% against the measured 69%.
- [x] Upgrade the detailed-proposal alignment receipt to schema v2 and bind it
  to both the reviewed private proposal SHA-256 and the exact aggregate-evidence
  bytes/schema. The CI artifact audit now deterministically recomputes the full
  receipt and rejects evidence drift, malformed structures, wrong proposal
  identity, or altered scores. The complete local suite passes 253 tests with
  69% branch-aware coverage, and all compile, lint, type, frozen-evaluator,
  application-risk, and Moshi-risk checks pass.
- [x] Harden the CLI-exposed batch re-ASR pipeline with deterministic tests for
  explicit service discovery, NeMo preparation rejection/success, verbatim
  Persian normalization, HTTP/Whisper fallback selection, failed probes,
  per-row failures, dry-run behavior, limits, resume semantics, parallel local
  teachers, and serialized initialized SpeechService use. The module rises from
  0% to 78% branch-aware coverage; the complete suite passes 261 tests at
  72.06%, and the enforced CI floor rises conservatively from 68% to 71%.
- [x] Harden legacy multi-channel ingest and YouTube preparation with
  deterministic local/remote, filtering, error, MP3 fallback, inventory, and
  cleanup tests. Correct unsafe episode-level resume so an interrupted episode
  resumes by utterance instead of silently losing remaining rows; preserve the
  inventory's channel-qualified split rather than re-hashing a bare filename;
  always remove temporary remote downloads; and emit timezone-aware completion
  timestamps. Ingest coverage rises from 18% to 94%, preparation from 32% to
  96%, and the complete suite passes 271 tests at 75.38%, allowing the enforced
  floor to rise conservatively from 71% to 74%.
- [x] Harden the automatic evaluation benchmark runner across capped-H100 and
  eligible-hardware paths while preserving the rule that server/synthetic
  timings can never become official browser E2E evidence. Test synthetic versus
  recorded detector scopes, successful and failed ASR-on-TTS diagnostics, and
  summary claim boundaries. Close every temporary WAV descriptor, remove files
  even when inference fails, serialize unavailable WER as standards-compliant
  JSON `null` rather than `NaN`, create summary parents, and remove a redundant
  report write. Benchmark coverage rises from 16% to 100%; the complete suite
  passes 278 tests at 76.49%, allowing the enforced floor to rise from 74% to
  75%.
- [x] Harden scientific dependency drift detection against vulnerability-feed
  primary-ID migrations. The 2026-09-10 feed changed five reviewed Torch CVE
  identifiers to PYSEC identifiers while retaining the CVEs as aliases; compare
  full CVE/PYSEC/GHSA identity groups, merge duplicate records, fail on duplicate
  package-version ambiguity, and retain strict failure for every genuinely
  unmatched advisory. Focused policy tests and the live 56-identity pinned
  Moshi exception audit pass without broadening the accepted-risk policy.
- [x] Confirm the advisory-identity correction in complete GitHub Actions run
  `34488968607`: all compile, lint, type, history/privacy, frozen-source,
  application-risk, 280-test/coverage, and pinned Moshi risk-drift stages pass.
- [x] Correct the historical codec/component survey's mislabeled `mulaw8`
  calculation: it now applies a real 256-level mu-law quantization bottleneck,
  changing only that diagnostic SNR from an invalid identity-like 99.33 dB to
  37.81 dB. Relabel mel/Griffin-Lim and Encodec so neither implies evidence
  about CosyVoice2; make import-only probes explicit; return JSON-safe `null`
  for undefined SNR; clean temporary ASR files even on write failure; and make
  persisted paths repository-relative. The independent Encodec CER remains
  0.087; its automatic transcript-content proxy is no longer mislabeled as
  phonetic preservation, and no model-selection conclusion changes. The
  complete local suite passes 285 tests at 77.55% branch-aware coverage,
  allowing the enforced floor to rise from 75% to 76%.
- [x] Harden the CLI-exposed data factory against stale and partial state.
  Caption promotion now verifies a non-empty replacement before archiving the
  active mix, preserves the old mix if any copy fails, and uses same-filesystem
  atomic replacement. Missing current input can no longer diarize a stale
  filtered manifest or report stale filtered hours as current. JSONL duration
  counting and evidence-report loading fail closed on malformed/non-object,
  negative, or non-finite data instead of silently undercounting. Isolated
  success/failure regressions touch no real corpus. The complete suite passes
  293 tests at 78.43% branch-aware coverage; factory coverage rises from 51%
  to 89%, allowing the enforced floor to rise from 76% to 77%.
- [x] Confirm that factory hardening in complete GitHub Actions run
  `34681925809`, including all history/privacy, evaluator-binding, dependency,
  test, coverage, and scientific-risk gates.
- [x] Make caption filtering and acoustic-quality decisions fail closed.
  Reject malformed/non-object manifest rows, invalid threshold relationships,
  and non-finite/negative durations before replacing any output; reject
  non-finite supplied SNR; emit a finite 0 dB proxy for clips too short to
  estimate an energy distribution; pack rows only when they fit the configured
  maximum instead of overshooting it; and atomically flush/fsync/replace the
  filtered JSONL with standards-compliant finite JSON. The complete suite
  passes 301 tests at 78.625% branch-aware coverage, allowing the enforced
  floor to rise from 77% to 78%.
- [x] Confirm the filtering hardening in complete GitHub Actions run
  `34682161682`, including all history/privacy, evaluator-binding, dependency,
  test, coverage, and scientific-risk gates.
- [x] Harden future interruption-detector evidence without changing the frozen
  proxy. Recorded audio/features now fail closed on malformed rows, missing
  referenced audio, corrupt or non-finite feature vectors, unknown labels, and
  absent speaker/session groups rather than silently dropping or leakage-prone
  grouping. Clean negatives cover every adjacent speaker boundary; threshold
  and grouped-bootstrap inputs are validated; and the proposal criterion is
  implemented as accuracy strictly greater than 80%. Every training report now
  states that metadata labels are not independent human review, that
  `human_verified_labels=0`, and that official detector eligibility/acceptance
  is false. The frozen 132-event result remains byte-for-byte untouched and is
  still only an automatic-label proxy. The complete local suite passes 306
  tests at 78.620% branch-aware coverage; the enforced floor remains 78% to
  retain a non-flaky margin.
- [x] Confirm that detector-evidence hardening in complete GitHub Actions run
  `34682630400`, including all compile, lint, type, history/privacy,
  evaluator-binding, dependency, 306-test/coverage, and scientific-risk gates.
- [x] Make the classical detector's fit/runtime artifact fail closed. Reject
  empty or mismatched frame supervision, wrong-width/non-finite features,
  non-integral/out-of-schema labels, and single-class training; never classify
  empty input as an interruption; validate evaluation/mask alignment; train a
  replacement pipeline before mutating the live detector; refuse to save an
  unfitted model; and fsync/atomically replace a versioned model payload.
  Loading is explicitly trusted-local only and validates fitted component
  types plus the exact feature width while remaining compatible with both
  existing legacy detector files (90/90 features). The complete local suite
  passes 314 tests at 78.895% branch-aware coverage and the core detector rises
  to 92.92%; the floor remains 78% because the exact result is below 79%.
- [x] Confirm detector-model hardening in complete GitHub Actions run
  `34683046737`, including every compile, lint, type, history/privacy,
  evaluator-binding, dependency, 314-test/coverage, and scientific-risk gate.
- [x] Harden all S3/rclone entry points without accessing or regenerating remote
  data. Apply a positive finite, configurable one-hour timeout to inventory and
  episode copies; reject malformed named remotes, endpoints containing URL
  credentials/query/fragment, ambiguous bucket names, and provider strings;
  label inventory status explicitly; write finite JSON with fsync and atomic
  replacement; and make any partial listing exit nonzero while recording only
  redacted error type/return code rather than raw stderr. Existing tracked S3
  evidence remains untouched. The complete local suite passes 320 tests at
  79.224% branch-aware coverage and S3 inventory rises from 57% to 87.43%. The
  floor remains 78% because raising it to 79% would leave only 0.224 points of
  non-flaky margin.
- [x] Confirm S3/rclone hardening in complete GitHub Actions run `34683424361`,
  including every compile, lint, type, history/privacy, evaluator-binding,
  dependency, 320-test/coverage, and scientific-risk gate.
- [x] Make cascade operation and generic benchmark claims fail closed without
  touching frozen results. Reject malformed or credential-bearing ASR URLs,
  invalid timeout/audio/sample-rate inputs, and malformed/non-object/oversized
  ASR responses; never invent “سلام” when ASR fails; label the rule-based
  latency helper as an unofficial component diagnostic; bind cached default
  Qwen loading to exact revision `cdbee75f17c01a7cc42f958dc650907174af0554`;
  reject Latin or over-25-word prompt-v2 generations; redact internal ASR/model
  errors from WebSocket clients; and expose `validated_cascade_ready` separately
  from transport liveness. The generic detector benchmark now preserves
  `official_detector_eligible=false` for recorded automatic labels instead of
  overwriting it with true. The complete local suite passes 327 tests at
  79.640% branch-aware coverage; cascade rises from 72% to 88.10%, benchmark
  coverage remains 100%, and the enforced floor rises from 78% to 79%.
- [x] Confirm cascade hardening in complete GitHub Actions run `34684242098`,
  including compile, lint, type, history/privacy, evaluator-binding,
  dependency, 327-test/coverage, and scientific-risk gates.
- [x] Harden audio and TTS I/O without regenerating frozen evidence. Reject
  unsupported, complex, boolean, non-finite, empty, and ambiguous channels-first
  audio; validate every sample rate; remove float-as-integer-PCM guessing; round
  PCM quantization; and fsync/atomically replace WAV output. Bound ffmpeg and
  Piper subprocesses with configurable positive finite timeouts. Make the
  formant diagnostic deterministic across processes, honor only an explicit
  Piper voice or the exact Mana filename, validate rendered WAV/audio, and make
  health readiness depend on a successful finite-audio render rather than model
  presence. The complete local suite passes 339 tests at 80.057% branch-aware
  coverage; audio and TTS reach 86.79% and 81.54%. The CI floor remains 79%
  because the 0.057-point margin above 80% is intentionally not treated as a
  stable cross-interpreter enforcement margin.
- [x] Confirm audio/TTS hardening in complete GitHub Actions run `34684905399`,
  including compile, lint, type, history/privacy, evaluator-binding,
  dependency, 339-test/coverage, and scientific-risk gates.
- [x] Make future measurement output fail closed without changing frozen
  evidence. Reject binary labels that would be truncated/coerced, invalid
  percentile/bootstrap controls, non-numeric/negative latency or VRAM, and a
  forced official flag when the actual samples are hardware-ineligible. Reject
  non-finite metric JSON and fsync/atomically replace reports while preserving
  an existing file on failure. The complete local suite passes 343 tests at
  80.167% branch-aware coverage, and metrics reaches 83.85%. The enforced floor
  remains 79% because 0.167 points is not a stable cross-interpreter margin.
- [x] Confirm measurement-evidence hardening in complete GitHub Actions run
  `34685451104`, including compile, lint, type, history/privacy,
  evaluator-binding, dependency, 343-test/coverage, and scientific-risk gates.
- [x] Make the legacy experimental Whisper/projector path unambiguously fail
  closed. Use restricted `weights_only` checkpoint loading and redact load
  errors; prevent even a fabricated `deployable_s2s_v1` metadata bundle from
  claiming a supported direct runtime; never replace missing manifest audio or
  a missing explicit manifest with synthetic speech; confine generated fixtures
  to `train-s2s-smoke`; validate schedule/dataset controls; and reject the
  unimplemented Encodec option instead of mislabeling log-mel reconstruction.
  The complete local suite passes 346 tests at 80.088% branch-aware coverage.
- [x] Confirm direct-path fail-closed hardening in complete GitHub Actions run
  `34685848406`, including compile, lint, type, history/privacy,
  evaluator-binding, dependency, 346-test/coverage, and scientific-risk gates.
- [x] Bound every active CI/release-build subprocess without altering frozen
  evaluator implementations or multi-hour systemd-managed training launchers.
  Full-history and historical-source Git reads use a validated configurable
  120-second timeout; the pinned browser build uses a separate validated
  30-minute timeout. Make generated evidence Markdown fsync/atomic and require
  CI to reproduce it exactly from authoritative JSON. The proposal and frozen
  evaluator receipts remain byte-identical. The complete local suite passes 351
  tests at 80.159% branch-aware coverage; evidence aggregation reaches 97.35%,
  and the CI floor remains 79% because the cross-interpreter margin over 80%
  remains too narrow.
- [x] Confirm the bounded evidence and source-receipt changes in complete GitHub
  Actions run `34687281188` at `72d531a`.
- [x] Harden consented-session persistence and official study evidence. Bind
  participant/retention/hardware provenance across reconnects; reject invalid
  numeric/audio/boolean inputs; serialize turn and rating JSONL updates under
  inter-process locks with finite JSON, fsync, and atomic replacement; clean up
  orphan audio after failed persistence; make manifest and study-summary output
  atomic; remove absolute server paths from rating responses; and make invalid
  stored turn rows fail the canonical study gate. The complete local suite
  passes 358 tests at 80.323% branch-aware coverage, with the recorder at
  90.35%; the CI floor remains 79%.
- [x] Confirm that recorder hardening in complete GitHub Actions run
  `34688277409` at `a02363a`; all remote gates passed.
- [x] Run a security-specific Ruff audit beyond the default rules. Remove all
  optimization-sensitive assertions and implicit security-MD5 warnings from
  production source; use explicit fail-closed invariant errors and declare the
  deterministic episode split hash as non-security. Enforce `S101,S324` for
  `src` in CI. All 358 local tests pass at 80.222% branch-aware coverage.
- [x] Confirm the expanded production security gate in complete GitHub Actions
  run `34699895879` at `d840471`; all remote gates passed.
- [x] Make the human-study ingestion boundary strictly typed and fail closed.
  Reject string/number coercion into consent, boolean or string Likert values,
  unsafe session/speaker IDs, invalid age/detector/prompt/interrupt enums, and
  unknown rating fields before persistence. Validate WebSocket metadata as a
  complete typed state and preserve its last valid value after rejected input.
  The complete local suite passes 359 tests at 80.256% branch-aware coverage;
  targeted recording, transport, and persistence tests pass 21/21.

Exit: green audit, reproducible runtime/adapter, traceable thesis tables,
privacy-safe repository at approved visibility, restricted handoff, immutable tag.

## Formal requirement traceability

| Requirement | Conservative acceptance test | Current state | Final evidence |
|---|---|---|---|
| Working Persian S2S prototype | Frozen system accepts audited user speech and emits Persian-script, non-silent speech on group-disjoint inputs; relevance/coherence uses a predeclared automatic rubric | **Passed 40-row automatic semantic final test:** relevance 3.150 vs 1.425, coherence 3.325, all seven gates; also passed 9/9 mechanics rows and round-trip CER 7.14% / WER 30.41%. Human-perceptual claims remain pending | Qwen4B prompt-v2 protocol/development/final reports; cascade mechanics/intelligibility reports |
| Full duplex | Mic remains active; interruption stops playback and becomes next-turn context | Server transport continuity passed; physical-browser source-stop/ack trace and direct model remain pending | WebSocket continuation tests; future client traces |
| End-to-end ≤500 ms | Max `T_first_audio` ≤500 ms unless another statistic is predeclared; always p50/p95/max | Pending | Physical-4090 `official_e2e` telemetry |
| Open base adapted to Persian | Moshika 7B LoRA trained on the waiver-bound Persian response pairs | Training complete through v6.2. V6.2 shows a 32.20% in-sample text-loss reduction but 0/9 runtime rows for every checkpoint; no adapter is promoted and all later final tests remain untouched | Config, logs, adapter hashes, reevaluation/runtime reports |
| 100–200 conversational hours | Final audited/exported hours in range; group-clean, no reused intervals | Frozen trained derivative: **108.584 exported h**, 6,754/6,754 pairs. Recommended balanced v2: **110.374 source-pair h**, 6,551 pairs, four channels, 54.0% maximum share. Both have zero reuse/leaks; v2 was not used by completed runs | Conversation audit/export report; balanced-v2 selection/balance/structural audits |
| Noise/overlap/interruption labels | Conditions present, QA complete, verified interruption, agreed precision | Automatic only | QA reports and final counts |
| Classical detector >80% | Independently labeled real group-held-out event accuracy >80%, F1/FAR/FRR reported | Recorded-audio automatic-label proxy: 81.06% on 132 events / 22 sessions, but CI crosses 80% and official eligibility is false | Independently labeled held-out report/hash |
| 12–24 GB evaluation | Full model fits/runs officially on physical 4090 | Pending | Preflight, VRAM, telemetry |
| Human evaluation | 5–10 Persian speakers, ≥2 aged 60+, complete ratings | Pending | Study summary/analysis |
| Elderly findings | Evidence from ≥2 aged 60+, cautiously interpreted | Pending | Age-stratified results |
| Documented code/dataset | Reproducible code/metadata at approved visibility; restricted corpus handled per approval | Implemented, Apache-2.0 licensed for project-owned source/documentation, with final full/concise report PDFs and sources included under `thesis-report/`; the adapted template retains CC-BY-4.0 attribution and restricted artifacts retain separate terms | Final repo/provenance/handoff/report audit |

## Inputs required from the student

1. If protected `main` is required, authorize an eligible GitHub plan or a
   visibility change.
2. Provide the physical 4090 if it becomes available before the submission
   freeze; otherwise report the official hardware/latency gate as unavailable.
3. Arrange the consented 5–10-person study with at least two participants aged
   60+, or obtain a written scope amendment.
4. Optional strict-path recovery only: complete the preserved 40-row and 24-row
   listening sheets. They are deliberately waived—not fabricated—for the active
   limited run and do not block its export/probe/training path.

Continue automatically through engineering-owned work. Never backfill human,
4090, detector, or study evidence after seeing an outcome.
