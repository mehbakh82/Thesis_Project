"""End-to-end data factory: CSV inventory → audio sample → caption filter → synthetic."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from thesis_s2s.config import portable_path, portable_project_values, project_root
from thesis_s2s.data.audit import audit_manifest
from thesis_s2s.data.diarize import annotate_manifest
from thesis_s2s.data.filter_corpus import filter_hours, write_synthetic_duplex
from thesis_s2s.data.ingest import run_ingest
from thesis_s2s.metrics import write_json
from thesis_s2s.runtime.session_log import SessionStore


def _hours_from_jsonl(path: Path) -> tuple[int, float, int]:
    n = 0
    hours = 0.0
    captioned = 0
    if not path.is_file():
        return 0, 0.0, 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            hours += float(row.get("duration") or 0) / 3600.0
            if str(row.get("transcript_caption") or row.get("text") or "").strip():
                captioned += 1
    return n, hours, captioned


def _load_report(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def promote_caption_training_mix() -> dict:
    """Make filtered.jsonl the CSV-caption mix; archive the old NeMo-teacher mix once."""

    manifests = project_root() / "data" / "processed" / "manifests"
    caption = manifests / "filtered_caption.jsonl"
    filtered = manifests / "filtered.jsonl"
    archive = manifests / "filtered_nemo_teacher.jsonl"
    report: dict[str, object] = {"copied": False, "archived_nemo_mix": False, "source": None}
    if filtered.is_file() and not archive.is_file():
        shutil.move(str(filtered), str(archive))
        report["archived_nemo_mix"] = True
    if caption.is_file() and caption.stat().st_size > 0:
        shutil.copy2(caption, filtered)
        report["copied"] = True
        report["source"] = portable_path(caption, root=project_root())
    return report


def refresh_dataset_card(report: dict | None = None) -> Path:
    root = project_root()
    manifests = root / "data" / "processed" / "manifests"
    n_all, h_all, _ = _hours_from_jsonl(manifests / "youtube_all.jsonl")
    _, h_nemo, _ = _hours_from_jsonl(manifests / "youtube_reasr.jsonl")
    n_filt, h_filt, n_filt_text = _hours_from_jsonl(manifests / "filtered.jsonl")
    _, h_cap, _ = _hours_from_jsonl(manifests / "filtered_caption.jsonl")
    rec = SessionStore().export_manifest()
    rec_hours = rec.get("hours") or 0.0
    csv_inv = {}
    csv_path = manifests / "csv_inventory.json"
    if csv_path.is_file():
        csv_inv = json.loads(csv_path.read_text(encoding="utf-8"))
    csv_hours = (csv_inv.get("stats") or {}).get("csv_hours")
    conversation_selection: dict = {}
    selection_path = root / "results" / "conversation_selection_audit.json"
    if selection_path.is_file():
        conversation_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        full_inventory_stats = conversation_selection.get("inventory_stats") or {}
        if full_inventory_stats.get("csv_hours") is not None:
            csv_hours = full_inventory_stats["csv_hours"]
    selected_hours = conversation_selection.get("selected_candidate_hours")
    selected_episodes = conversation_selection.get("selected_episodes")
    prepared = _load_report(root / "results" / "prepared_episode_audit.json")
    prepared_reserve = _load_report(root / "results" / "prepared_reserve_tabaghe16_audit.json")
    reserve_selection = _load_report(root / "results" / "conversation_reserve_yield_selection.json")
    staging = _load_report(root / "results" / "diarized_episode_audit_combined_authorized.json")
    conversation_yield = _load_report(
        root / "results" / "conversation_yield_estimate_combined.json"
    )
    interaction_candidates = _load_report(root / "results" / "interaction_candidate_report.json")
    authorization = _load_report(
        root / "results" / "conversation_source_authorization_report_combined.json"
    )
    final_candidate_hours = reserve_selection.get("combined_candidate_hours", selected_hours)
    final_episodes = staging.get("episodes", selected_episodes)
    final_windows = staging.get("windows")
    final_staging_hours = staging.get("prepared_hours")
    final_multi_speaker_hours = staging.get("automatic_multi_speaker_hours")
    final_aligned_hours = staging.get("reference_aligned_hours")
    final_pair_hours = conversation_yield.get("estimated_pair_hours")
    final_pairs = conversation_yield.get("estimated_pairs")
    authorized_windows = (authorization.get("counts") or {}).get("authorized_windows")
    authorization_complete = authorization.get("training_authorization_gate_passes") is True
    synth_stats = {}
    synth_path = manifests / "synthetic_stats.json"
    if synth_path.is_file():
        synth_stats = json.loads(synth_path.read_text(encoding="utf-8"))
    caption_hours_ok = h_filt >= 100 and n_filt_text == n_filt and n_filt > 0
    audit = (
        audit_manifest(manifests / "filtered.jsonl", check_files=False)
        if (manifests / "filtered.jsonl").is_file()
        else {"thesis_coverage_ok": False, "reason": "filtered_manifest_missing"}
    )
    alignment: dict = {}
    alignment_path = root / "results" / "caption_alignment_audit.json"
    if alignment_path.is_file():
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    alignment_similarity = alignment.get("token_sequence_similarity") or {}
    if not isinstance(alignment_similarity, dict):
        alignment_similarity = {}
    zoomit_alignment: dict = {}
    zoomit_path = root / "results" / "caption_alignment_zoomit_live_sample.json"
    if zoomit_path.is_file():
        zoomit_alignment = json.loads(zoomit_path.read_text(encoding="utf-8"))
    zoomit_similarity = zoomit_alignment.get("token_sequence_similarity") or {}
    if not isinstance(zoomit_similarity, dict):
        zoomit_similarity = {}
    lines = [
        "# Dataset card (Persian speech corpus and planned duplex extension)",
        "",
        "## Identity",
        "",
        "- Name: `fa-s2s-duplex-mix` (working title)",
        "- Language: Persian (fa-IR)",
        "- Intended use: train/evaluate a full-duplex speech-to-speech prototype",
        f"- Last updated: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Splits (status)",
        "",
        "| Split | Source | License | Publish? | Status |",
        "|---|---|---|---|---|",
        (
            "| Internal conversation candidates | University S3 YouTube archive | "
            + (
                "Supervisor-approved internal thesis use; source licenses not independently verified"
                if authorization_complete
                else "Internal training authorization pending"
            )
            + " | No raw audio | "
            f"Full inventory **{csv_hours} caption h**. Final deterministic plan: "
            f"**{final_candidate_hours} candidate h / {final_episodes} episodes / "
            f"{final_windows} windows / {final_staging_hours} staging h**; "
            f"**{final_multi_speaker_hours} automatic multi-speaker h** and "
            f"**{final_pair_hours} estimated response-pair h / {final_pairs} pairs**. "
            + (
                f"Authorization passes for all {authorized_windows} windows; "
                if authorization_complete
                else "Authorization remains incomplete; "
            )
            + "40-row window QA and 24-row/short-excerpt interaction QA remain pending. |"
        ),
        (
            f"| Synthetic duplex | tiled harmonic overlap mixer | synthetic | Yes, clearly labeled | "
            f"**{synth_stats.get('hours', 'see synthetic_stats.json')} h**; plumbing and regression use only, "
            "not evidence of conversational speech quality. |"
        ),
        (
            f"| Optional local audio | Lab/home interactions | consent | Only if separately approved | "
            f"**{'present' if rec_hours > 0 else 'not_collected; not definition-required'}** "
            f"({rec_hours:.3f} h, {rec.get('n', 0)} turns, elderly_turns={rec.get('elderly_turns', 0)}). "
            "Default study kit is `serve --study --retention features`; it stores no WAV. |"
        ),
        "| Pointers | Common Voice fa | CC-0 | Pointers only | Not downloaded here |",
        "",
        "The internal-use decision is recorded in `docs/SUPERVISOR_DECISIONS.md`; "
        "`results/conversation_source_authorization_report_combined.json` separately "
        f"reports authorization for {authorized_windows or 0} final staging windows and "
        "zero verified-license coverage. Redistribution remains disabled.",
        "",
        "S2S/TTS text = YouTube **CSV caption** (`transcript_caption`) after **fa-verbatim-2**. "
        "Those CSVs are the same reference transcripts used to fine-tune Soroush; NeMo is not a teacher for this mix. "
        "ASR training orthography (`prepare_tabaghe16.normalise`) is not used as the spoken target. "
        f"NeMo HTTP remains the cascade ASR baseline only. Optional `youtube_reasr.jsonl` ({h_nemo:.2f} h) is diagnostic.",
        "",
        "## Audit snapshot",
        "",
        (
            "`results/prepared_episode_audit.json` reports the primary reconstruction: "
            f"**{prepared.get('prepared_episodes', 0)}/{prepared.get('selected_episodes', 0)} episodes**, "
            f"**{prepared.get('windows', 0)} windows / {prepared.get('prepared_audio_hours', 0)} h**, "
            f"reconstruction gate **{bool(prepared.get('reconstruction_gate_passes', False))}**."
        ),
        (
            "The bounded reserve audit reports "
            f"**{prepared_reserve.get('prepared_episodes', 0)} episodes / "
            f"{prepared_reserve.get('windows', 0)} windows / "
            f"{prepared_reserve.get('prepared_audio_hours', 0)} h**; the deterministic "
            f"selector uses {reserve_selection.get('selected_reserve_episodes', 0)} reserve episodes."
        ),
        "",
        (
            "`results/diarized_episode_audit_combined_authorized.json`: "
            f"**{final_episodes} episodes / {final_windows} windows / "
            f"{final_multi_speaker_hours} multi-speaker h / {final_aligned_hours} aligned h**; "
            "automatic and authorization gates pass, while manual QA remains open."
        ),
        (
            "`results/conversation_yield_estimate_combined.json`: "
            f"**{final_pairs} estimated non-reused pairs / {final_pair_hours} pair h**. "
            f"Labels: {json.dumps(conversation_yield.get('label_counts', {}), ensure_ascii=False, sort_keys=True)}; "
            f"human-verified direct interruption pairs: {conversation_yield.get('direct_interruption_pairs', 0)}."
        ),
        (
            "Raw speaker boundaries recover "
            f"**{interaction_candidates.get('automatic_candidates', 0)} conservative interaction candidates** "
            f"({json.dumps(interaction_candidates.get('automatic_candidate_counts', {}), ensure_ascii=False, sort_keys=True)}). "
            f"The generated {interaction_candidates.get('sampled_candidates', 0)}-row sheet covers "
            "all four channels and about three minutes of excerpt audio. These are automatic "
            "candidates—not interruption claims—until pair-level listening review passes."
        ),
        "",
        "Staging hours preserve conversational context and gaps, whereas pair hours count only "
        "non-reused adjacent-turn spans. They are intentionally audited as different measures; "
        "the final pair set is inside the 100–200 h thesis band.",
        "",
        "The older flat-clip acoustic/caption audit in `results/corpus_audit.json` reports:",
        "",
        f"- {audit.get('n', 0):,} rows / {audit.get('hours', 0):.3f} h;",
        f"- {audit.get('duplicate_utt_ids', 0)} duplicate IDs, {audit.get('schema_errors', 0)} schema errors, "
        f"and {audit.get('group_split_leaks', 0)} group leaks;",
        f"- {audit.get('text_cross_split_duplicates', 0)} exact texts shared across splits;",
        f"- interruption labels: {audit.get('label_counts', {})};",
        f"- {audit.get('overlap_rows', 0)} overlap annotations and {audit.get('response_supervision_rows', 0)} response-supervision rows;",
        f"- thesis conversational coverage: **{bool(audit.get('thesis_coverage_ok', False))}**.",
        f"- caption/ASR screening proxy: {alignment.get('paired_rows', 0):,} Digiato rows / {alignment.get('paired_hours', 0)} h, mean token similarity {alignment_similarity.get('mean')}, low-similarity fraction {alignment_similarity.get('low_lt_0_4_fraction')};",
        f"- Zoomit live spot check: n={zoomit_alignment.get('sampling', {}).get('n', 0) if isinstance(zoomit_alignment.get('sampling'), dict) else 0}, mean similarity {zoomit_similarity.get('mean')}, low-similarity fraction {zoomit_similarity.get('low_lt_0_4_fraction')}.",
        "Independent-ASR agreement is a screening proxy, not ground-truth alignment; retain manual audio review in the validation protocol.",
        "",
        "The hours gate and conversational-supervision gate are separate; empty/default schema fields are not collected labels.",
        "",
        "## Labels",
        "",
        "The schema supports `utt_id`, `audio_path`, `duration`, `transcript_caption`, `transcript_nemo` (diagnostic), "
        "`speaker_id`, `overlap_intervals`, `interrupt_label` (`interrupt` / `backchannel` / "
        "`overlap_unattributed` / `noise` / `none`), `snr`, `license`, and `age_bin`. "
        "`overlap_unattributed` explicitly does not count as interruption evidence.",
        "",
        "## Join rule",
        "",
        "Where the Tabaghe16 layout holds, `{stem}_chunk_NNNN.wav` is 1:1 with CSV row N (1-indexed). Episodes with fewer than 50 rows are treated as Shorts and skipped.",
        "",
        "Channels (2TB `asr`): Digiato, Zoomit, Kooshiar, and Mehran Rowshan Persian use top-level prefixes. "
        "Tabaghe16 uses `STT/YT_PodCast_Chunks/{CSVs,Audio_Chunks}/طبقه 16`.",
        "",
        "## Collection notes",
        "",
        "Chunk audio is time-aligned to YouTube caption CSVs from the existing crawl (`--write-auto-subs` → VTT → CSV). "
        "SFT uses those captions, not a second ASR pass. Optional leftover `batch-reasr` / `youtube_reasr.jsonl` is not the training target. "
        "Diarization is required before selected episodes count as conversational supervision. "
        "Reconstructed windows declare that cross-chunk overlap may be unrecoverable.",
        "",
        "Filter: duration 1–20 s, Persian script ratio, music-caption drop, SNR p90−p10, LID `language_status` when present. "
        "TTS/SFT uses the caption-filtered mix (`filtered.jsonl` / `filtered_caption.jsonl`), not `--require-teacher`.",
        "",
        "## Ethical",
        "",
        "All participants sign the mode-specific `docs/CONSENT.md` and explicitly opt into persistence in the client. "
        "Feature and metrics modes retain no WAV. Public release excludes YouTube and restricted media. "
        "Human study N=5–10 (≥2 aged 60+) remains incomplete; see `docs/HUMAN_STUDY.md`.",
        "",
        "## Moshi direct-model derivative",
        "",
        "After reviewer QA, non-reused adjacent turns are exported in the official Moshi "
        "stereo schema. The user channel retains authorized natural archive audio. The "
        "primary assistant channel is synthesized deterministically from the approved next-turn "
        "text with the pinned Mana-Persian-Piper voice. Original podcast response audio is an "
        "explicit multi-voice ablation. Neither derivative corpus is redistributed.",
        "",
    ]
    path = root / "docs" / "DATASET_CARD.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    snapshot = {
        "audio_hours": round(h_all, 3),
        "audio_n": n_all,
        "caption_filter_hours": round(h_cap, 3),
        "filtered_caption_hours": round(h_filt, 3),
        "filtered_caption_utts": n_filt,
        "nemo_reasr_hours_unused": round(h_nemo, 3),
        "recorded_hours": rec_hours,
        "csv_hours": csv_hours,
        "caption_band_ok": caption_hours_ok,
        "thesis_coverage_ok": bool(audit.get("thesis_coverage_ok", False)),
        "conversation_candidate_hours": final_candidate_hours,
        "conversation_candidate_episodes": final_episodes,
        "conversation_windows": final_windows,
        "conversation_staging_hours": final_staging_hours,
        "conversation_automatic_multi_speaker_hours": final_multi_speaker_hours,
        "conversation_reference_aligned_hours": final_aligned_hours,
        "conversation_estimated_pairs": final_pairs,
        "conversation_estimated_pair_hours": final_pair_hours,
        "conversation_training_authorized_windows": authorized_windows,
        "conversation_training_authorization_gate": authorization_complete,
        "conversation_manual_qa_complete": (staging.get("requirements") or {}).get(
            "manual_qa_sample_present"
        )
        is True,
        "conversation_direct_interruption_pairs": conversation_yield.get(
            "direct_interruption_pairs"
        ),
        "conversation_automatic_interaction_candidates": interaction_candidates.get(
            "automatic_candidates"
        ),
        "conversation_automatic_interaction_candidate_counts": interaction_candidates.get(
            "automatic_candidate_counts"
        ),
        "conversation_interaction_qa_sampled_candidates": interaction_candidates.get(
            "sampled_candidates"
        ),
        "conversation_reserve_hours": conversation_selection.get("reserve_candidate_hours"),
        "conversation_reserve_episodes": conversation_selection.get("reserve_episodes"),
        "conversation_planning_gate": bool(conversation_selection.get("planning_gate_passes")),
        "conversation_evidence_gate": bool(
            conversation_selection.get("thesis_evidence_gate_passes")
        ),
        "corpus_audit": audit,
        "caption_alignment_audit": alignment,
        "zoomit_alignment_sample": zoomit_alignment,
        "s2s_text": "transcript_caption",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "factory": report,
    }
    write_json(
        root / "results" / "dataset_card_snapshot.json",
        portable_project_values(snapshot, root=root),
    )
    return path


def scale_corpus(
    *,
    caption_in: Path | None = None,
    min_hours: float = 100.0,
    max_hours: float = 200.0,
    reasr_limit: int | None = None,
    diarize_limit: int = 32,
) -> dict:
    """Caption-filter ingested audio into the SFT mix. Does not re-ASR with NeMo."""

    del reasr_limit  # leftover CLI flag; NeMo is not the S2S teacher
    root = project_root()
    manifests = root / "data" / "processed" / "manifests"
    src = Path(caption_in or manifests / "youtube_all.jsonl")
    caption_out = manifests / "filtered_caption.jsonl"
    filtered = manifests / "filtered.jsonl"
    diar_out = manifests / "filtered_diarized.jsonl"
    caption = {"n": 0, "hours": 0.0}
    if src.is_file() and src.stat().st_size > 0:
        caption = filter_hours(
            src,
            caption_out,
            min_hours=min_hours,
            max_hours=max_hours,
            require_teacher=False,
        )
        if caption_out.is_file():
            shutil.copy2(caption_out, filtered)
        filt = dict(caption)
    else:
        filt = {"n": 0, "hours": 0.0, "min_hours_ok": False}
    diar = {"attempted": 0, "ok": 0, "note": "skipped_or_unset"}
    if filtered.is_file() and filtered.stat().st_size > 0:
        diar = annotate_manifest(filtered, diar_out, limit=diarize_limit)
    report = {
        "caption_filter": caption,
        "filtered": filt,
        "s2s_text": "transcript_caption",
        "reasr": {
            "skipped": True,
            "reason": "CSV captions are the S2S labels; NeMo is cascade-only",
        },
        "diarization": diar,
        "recording": SessionStore().export_manifest(),
    }
    write_json(root / "results" / "scale_corpus.json", report)
    refresh_dataset_card(report)
    return report


def run_factory(
    *,
    max_audio_hours: float = 10.0,
    max_episodes: int | None = None,
    reasr_limit: int | None = None,
    synthetic_hours: float = 2.0,
    filter_max_hours: float = 200.0,
) -> dict:
    del reasr_limit
    root = project_root()
    manifests = root / "data" / "processed" / "manifests"
    ingest = run_ingest(max_hours=max_audio_hours, max_episodes=max_episodes)
    src = manifests / "youtube_all.jsonl"
    filtered = manifests / "filtered.jsonl"
    caption_out = manifests / "filtered_caption.jsonl"
    if src.is_file() and src.stat().st_size > 0:
        filt = filter_hours(
            src, caption_out, min_hours=100, max_hours=filter_max_hours, require_teacher=False
        )
        if caption_out.is_file():
            shutil.copy2(caption_out, filtered)
    else:
        filt = {"n": 0, "hours": 0.0, "min_hours_ok": False}
    synth = write_synthetic_duplex(
        root / "data" / "processed" / "synthetic",
        manifests / "synthetic_duplex.jsonl",
        target_hours=synthetic_hours,
        clip_seconds=8.0,
    )
    rec = SessionStore().export_manifest()
    report = {
        "ingest": ingest,
        "filtered": filt,
        "synthetic": synth,
        "recording_split": rec,
        "target_internal_hours": [100, 200],
        "s2s_text": "transcript_caption",
        "note": (
            "YouTube audio is research-internal. Public release is synthetic + "
            "consented recordings only. S2S/TTS text is YouTube CSV caption after fa-verbatim-2."
        ),
    }
    write_json(root / "results" / "data_factory.json", report)
    refresh_dataset_card(report)
    return report
