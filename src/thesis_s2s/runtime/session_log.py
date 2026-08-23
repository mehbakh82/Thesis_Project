"""Session logger for consented recordings and the live MOS study."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_s2s import SAMPLE_RATE
from thesis_s2s.audio import write_wav
from thesis_s2s.bargein.features import FeatureConfig, context_vector, frame_feature_matrix
from thesis_s2s.config import project_root
from thesis_s2s.metrics import (
    binary_score_confidence_intervals,
    binary_scores,
    mean_confidence_interval,
)

PROMPTS = [
    {"id": "warmup_time", "fa": "ساعت چند است؟", "expect": "none"},
    {
        "id": "interrupt_story",
        "fa": "یک داستان بلند بگو، وسط آن حرفم را قطع می‌کنم.",
        "expect": "interrupt",
    },
    {
        "id": "backchannel",
        "fa": "حرف بزن؛ من فقط «آها» یا «بله» می‌گویم و نباید قطع شود.",
        "expect": "backchannel",
    },
    {
        "id": "noise",
        "fa": "در اتاق کمی نویز باشد (تلویزیون/ظرف‌ها) و یک سؤال بپرس.",
        "expect": "noise",
    },
    {"id": "free", "fa": "سه دقیقه گفت‌وگوی آزاد.", "expect": "none"},
]
VALID_PROMPTS = frozenset(prompt["id"] for prompt in PROMPTS)
VALID_LABELS = frozenset({"none", "interrupt", "backchannel", "noise"})
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


@dataclass
class SessionMeta:
    session_id: str
    speaker_id: str
    age_bin: str  # "under_60" | "60plus"
    consent: bool
    license: str = "consent"
    room: str = "quiet"
    notes: str = ""
    gpu_name: str = ""
    vram_gb: float | None = None
    retention: str = "audio"  # "audio" | "features" | "metrics"


@dataclass
class TurnRecord:
    utt_id: str
    prompt_id: str
    interrupt_label: str
    t_first_audio_ms: float | None = None
    server_generation_ms: float | None = None
    t_barge_in_ms: float | None = None
    stopped: bool = False
    duration: float = 0.0
    audio_filepath: str = ""
    interaction_audio_filepath: str = ""
    privacy_feature_vector: list[float] = field(default_factory=list)
    feature_schema: str | None = None
    retention: str = "audio"
    overlap_intervals: list = field(default_factory=list)


class SessionStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or project_root() / "data" / "recordings")
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        if not SAFE_ID.fullmatch(session_id):
            raise ValueError("session_id must use 1-64 ASCII letters, digits, '_' or '-'")
        path = self.root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def start(self, meta: SessionMeta) -> Path:
        if not meta.consent:
            raise PermissionError("explicit consent is required before study data are persisted")
        if not SAFE_ID.fullmatch(meta.speaker_id):
            raise ValueError("speaker_id must use 1-64 ASCII letters, digits, '_' or '-'")
        if meta.age_bin not in {"under_60", "60plus"}:
            raise ValueError("age_bin must be 'under_60' or '60plus'")
        if meta.retention not in {"audio", "features", "metrics"}:
            raise ValueError("retention must be 'audio', 'features', or 'metrics'")
        folder = self.session_dir(meta.session_id)
        meta_path = folder / "meta.json"
        if meta_path.is_file():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            identity = (
                previous.get("speaker_id"),
                previous.get("age_bin"),
                previous.get("retention", "audio"),
            )
            if identity != (meta.speaker_id, meta.age_bin, meta.retention):
                raise ValueError("existing session has different speaker, age group, or retention")
        else:
            meta_path.write_text(
                json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        (folder / "turns.jsonl").touch(exist_ok=True)
        return folder

    def add_turn(
        self,
        meta: SessionMeta,
        *,
        prompt_id: str,
        interrupt_label: str,
        user_audio: np.ndarray,
        interaction_audio: np.ndarray | None = None,
        t_first_audio_ms: float | None,
        t_barge_in_ms: float | None,
        stopped: bool,
        server_generation_ms: float | None = None,
    ) -> TurnRecord:
        if not meta.consent:
            raise PermissionError("explicit consent is required before study data are persisted")
        if prompt_id not in VALID_PROMPTS:
            raise ValueError(f"unknown prompt_id: {prompt_id}")
        if interrupt_label not in VALID_LABELS:
            raise ValueError(f"unknown interrupt_label: {interrupt_label}")
        folder = self.start(meta)
        n = (
            sum(1 for _ in (folder / "turns.jsonl").open("r", encoding="utf-8") if _.strip())
            if (folder / "turns.jsonl").is_file()
            else 0
        )
        utt_id = f"{meta.session_id}_{n:04d}"
        wav: Path | None = None
        interaction_wav: Path | None = None
        feature_vector: list[float] = []
        feature_schema: str | None = None
        interaction = np.asarray(
            interaction_audio if interaction_audio is not None else [], dtype=np.float32
        )
        if meta.retention == "audio":
            wav = folder / f"{utt_id}.wav"
            write_wav(wav, user_audio, SAMPLE_RATE)
            if interaction.size:
                interaction_wav = folder / f"{utt_id}_interaction.wav"
                write_wav(interaction_wav, interaction, SAMPLE_RATE)
        elif meta.retention == "features" and interaction.size:
            cfg = FeatureConfig()
            frames = frame_feature_matrix(interaction, cfg)
            aggregate = context_vector(frames, len(frames) - 1, cfg.context_frames)
            feature_vector = [round(float(value), 6) for value in aggregate]
            feature_schema = "energy-zcr-f0-voicing-mfcc13-delta13.context400.mean-std-max.v1"
        rec = TurnRecord(
            utt_id=utt_id,
            prompt_id=prompt_id,
            interrupt_label=interrupt_label,
            t_first_audio_ms=t_first_audio_ms,
            server_generation_ms=server_generation_ms,
            t_barge_in_ms=t_barge_in_ms,
            stopped=stopped,
            duration=round(len(user_audio) / SAMPLE_RATE, 3),
            audio_filepath=str(wav) if wav is not None else "",
            interaction_audio_filepath=str(interaction_wav) if interaction_wav is not None else "",
            privacy_feature_vector=feature_vector,
            feature_schema=feature_schema,
            retention=meta.retention,
        )
        with (folder / "turns.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        **asdict(rec),
                        "speaker_id": meta.speaker_id,
                        "age_bin": meta.age_bin,
                        "license": "consent",
                        "transcript_caption": None,
                        "transcript_nemo": None,
                        "feature_privacy_note": "lossy aggregate; no waveform retained"
                        if feature_vector
                        else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return rec

    def export_manifest(self, out_jsonl: Path | None = None) -> dict:
        out_jsonl = Path(
            out_jsonl or project_root() / "data" / "processed" / "manifests" / "recorded.jsonl"
        )
        n = 0
        hours = 0.0
        elderly = 0
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("w", encoding="utf-8") as dst:
            for turns in sorted(self.root.glob("*/turns.jsonl")):
                meta_path = turns.parent / "meta.json"
                meta = (
                    json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
                )
                for line in turns.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row.setdefault("age_bin", meta.get("age_bin"))
                    row.setdefault("gpu_name", meta.get("gpu_name"))
                    row.setdefault("vram_gb", meta.get("vram_gb"))
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                    hours += float(row.get("duration") or 0) / 3600.0
                    if row.get("age_bin") == "60plus":
                        elderly += 1
        report = {
            "n": n,
            "hours": round(hours, 3),
            "elderly_turns": elderly,
            "optional_local_audio_target_hours": [8, 15],
            "optional_target_met": hours >= 8,
            "definition_requires_local_audio_hours": False,
            "manifest": str(out_jsonl),
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }
        return report

    def study_summary(self, out_json: Path | None = None) -> dict:
        """Aggregate only consented live sessions and report readiness gates."""

        participants: set[str] = set()
        elderly: set[str] = set()
        first_audio: list[float] = []
        barge_in: list[float] = []
        ratings: list[dict] = []
        hardware_flags: list[bool] = []
        eligible_first_audio: list[float] = []
        eligible_barge_in: list[float] = []
        retention_counts: dict[str, int] = {"audio": 0, "features": 0, "metrics": 0}
        live_truth: list[int] = []
        live_pred: list[int] = []
        eligible_live_truth: list[int] = []
        eligible_live_pred: list[int] = []
        turns_n = 0
        for folder in sorted(path for path in self.root.iterdir() if path.is_dir()):
            meta_path = folder / "meta.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("consent"):
                continue
            speaker = str(meta.get("speaker_id") or folder.name)
            participants.add(speaker)
            if meta.get("age_bin") == "60plus":
                elderly.add(speaker)
            vram = meta.get("vram_gb")
            eligible_hardware = isinstance(vram, (int, float)) and 12 <= float(vram) <= 24
            hardware_flags.append(eligible_hardware)
            turns_path = folder / "turns.jsonl"
            if turns_path.is_file():
                for line in turns_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    turn = json.loads(line)
                    turns_n += 1
                    retention = str(turn.get("retention") or meta.get("retention") or "audio")
                    if retention in retention_counts:
                        retention_counts[retention] += 1
                    label = str(turn.get("interrupt_label") or "")
                    if label in VALID_LABELS:
                        live_truth.append(int(label == "interrupt"))
                        live_pred.append(int(turn.get("stopped") is True))
                        if eligible_hardware:
                            eligible_live_truth.append(int(label == "interrupt"))
                            eligible_live_pred.append(int(turn.get("stopped") is True))

                    if isinstance(turn.get("t_first_audio_ms"), (int, float)):
                        first_audio.append(float(turn["t_first_audio_ms"]))
                        if eligible_hardware:
                            eligible_first_audio.append(float(turn["t_first_audio_ms"]))
                    if isinstance(turn.get("t_barge_in_ms"), (int, float)):
                        barge_in.append(float(turn["t_barge_in_ms"]))
                        if eligible_hardware:
                            eligible_barge_in.append(float(turn["t_barge_in_ms"]))
            ratings_path = folder / "mos.jsonl"
            if ratings_path.is_file():
                ratings.extend(
                    json.loads(line)
                    for line in ratings_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )

        rating_means = {}
        rating_ci95 = {}
        required_rating_keys = ("naturalness", "latency", "interrupt_success", "satisfaction")
        complete_ratings = [
            row
            for row in ratings
            if all(isinstance(row.get(key), (int, float)) for key in required_rating_keys)
        ]
        complete_rating_speakers = {
            str(row.get("speaker_id")) for row in complete_ratings if row.get("speaker_id")
        }
        for key in required_rating_keys:
            values = [float(row[key]) for row in ratings if isinstance(row.get(key), (int, float))]
            rating_means[key] = round(float(np.mean(values)), 3) if values else None
            rating_ci95[key] = mean_confidence_interval(values)
        detector_reports = []
        for name in ("recorded_heldout_report.json", "feature_heldout_report.json"):
            path = project_root() / "results" / "bargein" / name
            if path.is_file():
                try:
                    detector_reports.append(json.loads(path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    continue
        detector_ready = any(
            report.get("evidence_class") in {"recorded_audio_heldout", "recorded_features_heldout"}
            and int(report.get("n") or (report.get("proposed") or {}).get("n") or 0) > 0
            and report.get("group_overlap") is False
            and (report.get("proposed") or {}).get("target_ok") is True
            for report in detector_reports
        )
        live_scores = binary_scores(live_truth, live_pred) if live_truth else None
        eligible_live_scores = (
            binary_scores(eligible_live_truth, eligible_live_pred) if eligible_live_truth else None
        )
        requirements = {
            "participants_5_to_10": 5 <= len(participants) <= 10,
            "elderly_participants_at_least_2": len(elderly) >= 2,
            "complete_ratings_cover_participants": bool(participants)
            and participants <= complete_rating_speakers,
            "eligible_client_first_audio_present": bool(eligible_first_audio),
            "eligible_client_barge_in_present": bool(eligible_barge_in),
            "physical_gpu_12_to_24_gb": any(hardware_flags),
            "real_heldout_detector_report": detector_ready,
        }
        report = {
            "status": "complete" if all(requirements.values()) else "not_collected_or_incomplete",
            "participants": len(participants),
            "elderly_participants": len(elderly),
            "turns": turns_n,
            "ratings": len(ratings),
            "complete_ratings": len(complete_ratings),
            "rating_means": rating_means,
            "rating_mean_ci95": rating_ci95,
            "t_first_audio_p50_ms": float(np.percentile(first_audio, 50)) if first_audio else None,
            "t_first_audio_p95_ms": float(np.percentile(first_audio, 95)) if first_audio else None,
            "t_first_audio_max_ms": max(first_audio) if first_audio else None,
            "t_barge_in_p95_ms": float(np.percentile(barge_in, 95)) if barge_in else None,
            "t_barge_in_max_ms": max(barge_in) if barge_in else None,
            "official_t_first_audio_p50_ms": float(np.percentile(eligible_first_audio, 50))
            if eligible_first_audio
            else None,
            "official_t_first_audio_p95_ms": float(np.percentile(eligible_first_audio, 95))
            if eligible_first_audio
            else None,
            "official_t_first_audio_max_ms": max(eligible_first_audio)
            if eligible_first_audio
            else None,
            "official_t_barge_in_p95_ms": float(np.percentile(eligible_barge_in, 95))
            if eligible_barge_in
            else None,
            "official_t_barge_in_max_ms": max(eligible_barge_in) if eligible_barge_in else None,
            "live_interrupt": live_scores.__dict__ if live_scores is not None else None,
            "live_interrupt_ci95": (
                binary_score_confidence_intervals(live_truth, live_pred) if live_truth else None
            ),
            "official_live_interrupt": (
                eligible_live_scores.__dict__ if eligible_live_scores is not None else None
            ),
            "official_live_interrupt_ci95": (
                binary_score_confidence_intervals(eligible_live_truth, eligible_live_pred)
                if eligible_live_truth
                else None
            ),
            "retention_counts": retention_counts,
            "requirements": requirements,
            "official_ready": all(requirements.values()),
            "raw_audio_required": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if out_json is not None:
            out_json = Path(out_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            out_json.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return report
