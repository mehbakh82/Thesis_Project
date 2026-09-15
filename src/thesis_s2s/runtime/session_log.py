"""Session logger for consented recordings and the live MOS study."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from thesis_s2s import (
    HUMAN_MOS_TARGET,
    SAMPLE_RATE,
    T_BARGE_IN_P95_MS,
    T_BARGE_IN_PROPOSAL_MAX_MS,
    T_FIRST_AUDIO_PROPOSAL_MAX_MS,
)
from thesis_s2s.audio import to_float32_mono, write_wav
from thesis_s2s.bargein.features import FeatureConfig, context_vector, frame_feature_matrix
from thesis_s2s.config import project_root
from thesis_s2s.metrics import (
    binary_score_confidence_intervals,
    binary_scores,
    mean_confidence_interval,
    meets_bargein_accuracy_target,
    write_json,
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
VALID_MEASUREMENT_SOURCES = frozenset({"unspecified", "live_browser"})
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _append_jsonl_atomic(path: Path, row_or_builder: dict | Callable[[int], dict]) -> dict:
    """Append one finite object under an inter-process lock via atomic replacement."""

    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = path.read_bytes() if path.is_file() else b""
        if existing and not existing.endswith(b"\n"):
            raise ValueError(f"refusing to append to truncated JSONL: {path}")
        rows = []
        for line_number, line in enumerate(existing.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL line {line_number} in {path}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            rows.append(parsed)
        row = row_or_builder(len(rows)) if callable(row_or_builder) else row_or_builder
        if not isinstance(row, dict):
            raise TypeError("JSONL row must be an object")
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(existing)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return row


def _optional_nonnegative_finite(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite, non-negative milliseconds or None")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite, non-negative milliseconds or None")
    return converted


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
    memory_capped: bool = False
    retention: str = "audio"  # "audio" | "features" | "metrics"
    measurement_source: str = "unspecified"  # "unspecified" | "live_browser"
    system_name: str = "unspecified"
    asr_backend: str = "unspecified"
    responder_model: str = "unspecified"
    responder_revision: str = "unspecified"
    responder_prompt_profile: str = "unspecified"
    tts_model_sha256: str = ""


@dataclass
class TurnRecord:
    utt_id: str
    prompt_id: str
    interrupt_label: str
    t_first_audio_ms: float | None = None
    server_generation_ms: float | None = None
    t_barge_in_ms: float | None = None
    client_playback_started_ack: bool = False
    client_playback_stopped_ack: bool = False
    stopped: bool = False
    talker: str = ""
    tts_backend: str = ""
    responder_backend: str = ""
    responder_fallback_used: bool | None = None
    asr_error: bool = False
    responder_error: bool = False
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
        if meta.measurement_source not in VALID_MEASUREMENT_SOURCES:
            raise ValueError("measurement_source must be 'unspecified' or 'live_browser'")
        for name in (
            "system_name",
            "asr_backend",
            "responder_model",
            "responder_revision",
            "responder_prompt_profile",
        ):
            value = getattr(meta, name)
            if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
                raise ValueError(f"{name} must be a non-empty single-line string")
        if meta.tts_model_sha256 and not SHA256.fullmatch(meta.tts_model_sha256):
            raise ValueError("tts_model_sha256 must be empty or a lowercase SHA-256 digest")
        if meta.vram_gb is not None and (
            isinstance(meta.vram_gb, bool)
            or not isinstance(meta.vram_gb, (int, float))
            or not math.isfinite(float(meta.vram_gb))
            or float(meta.vram_gb) <= 0
        ):
            raise ValueError("vram_gb must be finite, positive, or None")
        if not isinstance(meta.memory_capped, bool):
            raise ValueError("memory_capped must be a boolean")
        folder = self.session_dir(meta.session_id)
        meta_path = folder / "meta.json"
        import fcntl

        with (folder / ".session.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if meta_path.is_file():
                previous = json.loads(meta_path.read_text(encoding="utf-8"))
                identity = (
                    previous.get("speaker_id"),
                    previous.get("age_bin"),
                    previous.get("retention", "audio"),
                    previous.get("gpu_name", ""),
                    previous.get("vram_gb"),
                    previous.get("memory_capped", False),
                    previous.get("measurement_source", "unspecified"),
                    previous.get("system_name", "unspecified"),
                    previous.get("asr_backend", "unspecified"),
                    previous.get("responder_model", "unspecified"),
                    previous.get("responder_revision", "unspecified"),
                    previous.get("responder_prompt_profile", "unspecified"),
                    previous.get("tts_model_sha256", ""),
                )
                requested_identity = (
                    meta.speaker_id,
                    meta.age_bin,
                    meta.retention,
                    meta.gpu_name,
                    meta.vram_gb,
                    meta.memory_capped,
                    meta.measurement_source,
                    meta.system_name,
                    meta.asr_backend,
                    meta.responder_model,
                    meta.responder_revision,
                    meta.responder_prompt_profile,
                    meta.tts_model_sha256,
                )
                if identity != requested_identity:
                    raise ValueError(
                        "existing session has different participant, retention, or hardware provenance"
                    )
            else:
                write_json(meta_path, asdict(meta))
            (folder / "turns.jsonl").touch(exist_ok=True)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
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
        client_playback_started_ack: bool = False,
        client_playback_stopped_ack: bool = False,
        talker: str = "",
        tts_backend: str = "",
        responder_backend: str = "",
        responder_fallback_used: bool | None = None,
        asr_error: bool = False,
        responder_error: bool = False,
    ) -> TurnRecord:
        if not meta.consent:
            raise PermissionError("explicit consent is required before study data are persisted")
        if prompt_id not in VALID_PROMPTS:
            raise ValueError(f"unknown prompt_id: {prompt_id}")
        if interrupt_label not in VALID_LABELS:
            raise ValueError(f"unknown interrupt_label: {interrupt_label}")
        if not isinstance(stopped, bool):
            raise ValueError("stopped must be a boolean")
        if not isinstance(client_playback_started_ack, bool):
            raise ValueError("client_playback_started_ack must be a boolean")
        if not isinstance(client_playback_stopped_ack, bool):
            raise ValueError("client_playback_stopped_ack must be a boolean")
        for name, value in (
            ("talker", talker),
            ("tts_backend", tts_backend),
            ("responder_backend", responder_backend),
        ):
            if not isinstance(value, str) or "\n" in value or "\r" in value:
                raise ValueError(f"{name} must be a single-line string")
        if responder_fallback_used is not None and not isinstance(
            responder_fallback_used, bool
        ):
            raise ValueError("responder_fallback_used must be a boolean or None")
        if not isinstance(asr_error, bool) or not isinstance(responder_error, bool):
            raise ValueError("runtime error flags must be booleans")
        t_first_audio_ms = _optional_nonnegative_finite(t_first_audio_ms, name="t_first_audio_ms")
        t_barge_in_ms = _optional_nonnegative_finite(t_barge_in_ms, name="t_barge_in_ms")
        server_generation_ms = _optional_nonnegative_finite(
            server_generation_ms, name="server_generation_ms"
        )
        user = to_float32_mono(user_audio)
        if user.size == 0:
            raise ValueError("user_audio must be non-empty")
        folder = self.start(meta)
        wav: Path | None = None
        interaction_wav: Path | None = None
        feature_vector: list[float] = []
        feature_schema: str | None = None
        interaction = (
            to_float32_mono(interaction_audio)
            if interaction_audio is not None
            else np.zeros(0, dtype=np.float32)
        )
        if meta.retention == "features" and interaction.size:
            cfg = FeatureConfig()
            frames = frame_feature_matrix(interaction, cfg)
            aggregate = context_vector(frames, len(frames) - 1, cfg.context_frames)
            feature_vector = [round(float(value), 6) for value in aggregate]
            feature_schema = "energy-zcr-f0-voicing-mfcc13-delta13.context400.mean-std-max.v1"
        rec: TurnRecord | None = None

        def build_row(n: int) -> dict:
            nonlocal interaction_wav, rec, wav
            utt_id = f"{meta.session_id}_{n:04d}"
            if meta.retention == "audio":
                wav = folder / f"{utt_id}.wav"
                write_wav(wav, user, SAMPLE_RATE)
                if interaction.size:
                    interaction_wav = folder / f"{utt_id}_interaction.wav"
                    write_wav(interaction_wav, interaction, SAMPLE_RATE)
            rec = TurnRecord(
                utt_id=utt_id,
                prompt_id=prompt_id,
                interrupt_label=interrupt_label,
                t_first_audio_ms=t_first_audio_ms,
                server_generation_ms=server_generation_ms,
                t_barge_in_ms=t_barge_in_ms,
                client_playback_started_ack=client_playback_started_ack,
                client_playback_stopped_ack=client_playback_stopped_ack,
                stopped=stopped,
                talker=talker,
                tts_backend=tts_backend,
                responder_backend=responder_backend,
                responder_fallback_used=responder_fallback_used,
                asr_error=asr_error,
                responder_error=responder_error,
                duration=round(len(user) / SAMPLE_RATE, 3),
                audio_filepath=str(wav) if wav is not None else "",
                interaction_audio_filepath=str(interaction_wav) if interaction_wav else "",
                privacy_feature_vector=feature_vector,
                feature_schema=feature_schema,
                retention=meta.retention,
            )
            return {
                **asdict(rec),
                "speaker_id": meta.speaker_id,
                "age_bin": meta.age_bin,
                "license": "consent",
                "transcript_caption": None,
                "transcript_nemo": None,
                "feature_privacy_note": "lossy aggregate; no waveform retained"
                if feature_vector
                else None,
            }

        try:
            _append_jsonl_atomic(folder / "turns.jsonl", build_row)
        except Exception:
            for created in (interaction_wav, wav):
                if created is not None:
                    created.unlink(missing_ok=True)
            raise
        if rec is None:  # pragma: no cover - builder contract
            raise RuntimeError("turn row was not constructed")
        return rec

    def export_manifest(self, out_jsonl: Path | None = None) -> dict:
        out_jsonl = Path(
            out_jsonl or project_root() / "data" / "processed" / "manifests" / "recorded.jsonl"
        )
        n = 0
        hours = 0.0
        elderly = 0
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=out_jsonl.parent, prefix=f".{out_jsonl.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            dst = os.fdopen(descriptor, "w", encoding="utf-8")
            with dst:
                for turns in sorted(self.root.glob("*/turns.jsonl")):
                    meta_path = turns.parent / "meta.json"
                    meta = (
                        json.loads(meta_path.read_text(encoding="utf-8"))
                        if meta_path.is_file()
                        else {}
                    )
                    for line in turns.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        row.setdefault("age_bin", meta.get("age_bin"))
                        row.setdefault("gpu_name", meta.get("gpu_name"))
                        row.setdefault("vram_gb", meta.get("vram_gb"))
                        dst.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        n += 1
                        hours += float(row.get("duration") or 0) / 3600.0
                        if row.get("age_bin") == "60plus":
                            elderly += 1
                dst.flush()
                os.fsync(dst.fileno())
            os.replace(temporary, out_jsonl)
        finally:
            temporary.unlink(missing_ok=True)
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
        invalid_turn_rows = 0
        invalid_rating_rows = 0
        required_rating_keys = ("naturalness", "latency", "interrupt_success", "satisfaction")
        hardware_flags: list[bool] = []
        live_browser_flags: list[bool] = []
        system_provenance_flags: list[bool] = []
        system_identities: set[tuple[str, str, str, str, str, str]] = set()
        eligible_first_audio: list[float] = []
        eligible_barge_in: list[float] = []
        retention_counts: dict[str, int] = {"audio": 0, "features": 0, "metrics": 0}
        live_truth: list[int] = []
        live_pred: list[int] = []
        eligible_live_truth: list[int] = []
        eligible_live_pred: list[int] = []
        turns_n = 0
        valid_turns_n = 0
        valid_turn_speakers: set[str] = set()
        official_e2e_rows = 0
        official_interrupt_rows = 0
        missing_client_start_ack = 0
        missing_client_stop_ack = 0
        runtime_failure_rows = 0
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
            gpu_name = meta.get("gpu_name")
            eligible_hardware = bool(
                isinstance(vram, (int, float))
                and not isinstance(vram, bool)
                and math.isfinite(float(vram))
                and 12 <= float(vram) <= 24
                and isinstance(gpu_name, str)
                and gpu_name.strip()
                and meta.get("memory_capped") is False
            )
            hardware_flags.append(eligible_hardware)
            live_browser = meta.get("measurement_source") == "live_browser"
            live_browser_flags.append(live_browser)
            identity = (
                meta.get("system_name"),
                meta.get("asr_backend"),
                meta.get("responder_model"),
                meta.get("responder_revision"),
                meta.get("responder_prompt_profile"),
                meta.get("tts_model_sha256"),
            )
            system_provenance = bool(
                all(
                    isinstance(value, str) and value not in {"", "unspecified"}
                    for value in identity[:5]
                )
                and isinstance(identity[5], str)
                and SHA256.fullmatch(identity[5])
            )
            system_provenance_flags.append(system_provenance)
            if system_provenance:
                system_identities.add(identity)  # type: ignore[arg-type]
            eligible_session = eligible_hardware and live_browser and system_provenance
            turns_path = folder / "turns.jsonl"
            if turns_path.is_file():
                for line in turns_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        turn = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_turn_rows += 1
                        continue
                    turns_n += 1
                    if not isinstance(turn, dict):
                        invalid_turn_rows += 1
                        continue
                    retention = str(turn.get("retention") or meta.get("retention") or "audio")
                    label = turn.get("interrupt_label")
                    stopped = turn.get("stopped")
                    timing_values = (turn.get("t_first_audio_ms"), turn.get("t_barge_in_ms"))
                    acknowledgement_values = (
                        turn.get("client_playback_started_ack", False),
                        turn.get("client_playback_stopped_ack", False),
                    )
                    runtime_values = (
                        turn.get("talker", ""),
                        turn.get("tts_backend", ""),
                        turn.get("responder_backend", ""),
                    )
                    runtime_flags = (
                        turn.get("responder_fallback_used"),
                        turn.get("asr_error", False),
                        turn.get("responder_error", False),
                    )
                    valid_timings = all(
                        value is None
                        or (
                            isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                            and float(value) >= 0
                        )
                        for value in timing_values
                    )
                    valid_turn = (
                        retention in retention_counts
                        and label in VALID_LABELS
                        and isinstance(stopped, bool)
                        and valid_timings
                        and all(isinstance(value, bool) for value in acknowledgement_values)
                        and all(isinstance(value, str) for value in runtime_values)
                        and (
                            runtime_flags[0] is None
                            or isinstance(runtime_flags[0], bool)
                        )
                        and all(isinstance(value, bool) for value in runtime_flags[1:])
                    )
                    if not valid_turn:
                        invalid_turn_rows += 1
                        continue
                    retention_counts[retention] += 1
                    valid_turns_n += 1
                    valid_turn_speakers.add(speaker)
                    live_truth.append(int(label == "interrupt"))
                    live_pred.append(int(stopped is True))
                    if eligible_session:
                        official_e2e_rows += 1
                        eligible_live_truth.append(int(label == "interrupt"))
                        eligible_live_pred.append(int(stopped is True))
                        runtime_success = bool(
                            all(runtime_values)
                            and runtime_values[1] != "formant"
                            and runtime_values[2] != "rules"
                            and runtime_flags[0] is False
                            and runtime_flags[1] is False
                            and runtime_flags[2] is False
                        )
                        if not runtime_success:
                            runtime_failure_rows += 1

                    if turn.get("t_first_audio_ms") is not None:
                        first_audio.append(float(turn["t_first_audio_ms"]))
                    if eligible_session:
                        if (
                            turn.get("client_playback_started_ack") is True
                            and turn.get("t_first_audio_ms") is not None
                        ):
                            eligible_first_audio.append(float(turn["t_first_audio_ms"]))
                        else:
                            missing_client_start_ack += 1
                    if turn.get("t_barge_in_ms") is not None:
                        barge_in.append(float(turn["t_barge_in_ms"]))
                    if eligible_session and label == "interrupt":
                        official_interrupt_rows += 1
                        if (
                            turn.get("client_playback_stopped_ack") is True
                            and stopped is True
                            and turn.get("t_barge_in_ms") is not None
                        ):
                            eligible_barge_in.append(float(turn["t_barge_in_ms"]))
                        else:
                            missing_client_stop_ack += 1
            ratings_path = folder / "mos.jsonl"
            if ratings_path.is_file():
                for line in ratings_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_rating_rows += 1
                        continue
                    linked = (
                        isinstance(row, dict)
                        and row.get("consent") is True
                        and row.get("session_id") == folder.name
                        and row.get("speaker_id") == speaker
                        and row.get("age_bin") == meta.get("age_bin")
                    )
                    bounded = isinstance(row, dict) and all(
                        type(row.get(key)) is int and 1 <= row[key] <= 5
                        for key in required_rating_keys
                    )
                    if linked and bounded:
                        ratings.append(row)
                    else:
                        invalid_rating_rows += 1

        rating_means = {}
        rating_ci95 = {}
        complete_ratings = list(ratings)
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
            and meets_bargein_accuracy_target((report.get("proposed") or {}).get("accuracy"))
            and meets_bargein_accuracy_target(
                (report.get("proposed") or {}).get("interrupt_f1")
            )
            for report in detector_reports
        )
        live_scores = binary_scores(live_truth, live_pred) if live_truth else None
        eligible_live_scores = (
            binary_scores(eligible_live_truth, eligible_live_pred) if eligible_live_truth else None
        )
        all_sessions_physical = bool(hardware_flags) and all(hardware_flags)
        all_sessions_live_browser = bool(live_browser_flags) and all(live_browser_flags)
        system_provenance_complete = bool(system_provenance_flags) and all(
            system_provenance_flags
        )
        system_provenance_consistent = len(system_identities) == 1
        acknowledgements_complete = bool(official_e2e_rows) and bool(official_interrupt_rows)
        acknowledgements_complete = bool(
            acknowledgements_complete
            and missing_client_start_ack == 0
            and missing_client_stop_ack == 0
        )
        official_first_audio_max = max(eligible_first_audio) if eligible_first_audio else None
        official_barge_in_p95 = (
            float(np.percentile(eligible_barge_in, 95)) if eligible_barge_in else None
        )
        official_barge_in_max = max(eligible_barge_in) if eligible_barge_in else None
        official_latency_gate_passed = bool(
            acknowledgements_complete
            and system_provenance_complete
            and system_provenance_consistent
            and runtime_failure_rows == 0
            and official_first_audio_max is not None
            and official_first_audio_max <= T_FIRST_AUDIO_PROPOSAL_MAX_MS
        )
        official_barge_in_engineering_gate_passed = bool(
            acknowledgements_complete
            and system_provenance_complete
            and system_provenance_consistent
            and runtime_failure_rows == 0
            and official_barge_in_p95 is not None
            and official_barge_in_p95 <= T_BARGE_IN_P95_MS
        )
        official_interrupt_latency_le_150_ms = bool(
            acknowledgements_complete
            and system_provenance_complete
            and system_provenance_consistent
            and runtime_failure_rows == 0
            and official_barge_in_max is not None
            and official_barge_in_max <= T_BARGE_IN_PROPOSAL_MAX_MS
        )
        naturalness_mos = rating_means.get("naturalness")
        naturalness_mos_at_least_3_5 = bool(
            naturalness_mos is not None and naturalness_mos >= HUMAN_MOS_TARGET
        )
        official_e2e_eligible = bool(
            invalid_turn_rows == 0
            and all_sessions_physical
            and all_sessions_live_browser
            and system_provenance_complete
            and system_provenance_consistent
            and official_e2e_rows > 0
            and official_e2e_rows == valid_turns_n
            and acknowledgements_complete
            and runtime_failure_rows == 0
        )
        requirements = {
            "participants_5_to_10": 5 <= len(participants) <= 10,
            "elderly_participants_at_least_2": len(elderly) >= 2,
            "valid_turns_cover_participants": bool(participants)
            and participants <= valid_turn_speakers,
            "complete_ratings_cover_participants": bool(participants)
            and participants <= complete_rating_speakers,
            "turn_rows_valid": invalid_turn_rows == 0,
            "rating_rows_valid": invalid_rating_rows == 0,
            "naturalness_mos_at_least_3_5": naturalness_mos_at_least_3_5,
            "live_browser_measurement": all_sessions_live_browser,
            "system_provenance_complete_and_consistent": bool(
                system_provenance_complete and system_provenance_consistent
            ),
            "runtime_completed_without_fallback_or_error": bool(
                official_e2e_rows and runtime_failure_rows == 0
            ),
            "official_rows_cover_all_valid_turns": bool(valid_turns_n)
            and official_e2e_rows == valid_turns_n,
            "client_playback_acknowledgements_complete": acknowledgements_complete,
            "eligible_client_first_audio_present": bool(eligible_first_audio),
            "eligible_client_barge_in_present": bool(eligible_barge_in),
            "first_audio_max_le_500_ms": official_latency_gate_passed,
            "barge_in_p95_le_300_ms_engineering": official_barge_in_engineering_gate_passed,
            "interrupt_latency_max_le_150_ms_proposal": (
                official_interrupt_latency_le_150_ms
            ),
            "physical_gpu_12_to_24_gb": all_sessions_physical,
            "real_heldout_detector_report": detector_ready,
        }
        report = {
            "schema_version": 2,
            "evidence_class": "official_e2e",
            "status": "complete" if all(requirements.values()) else "not_collected_or_incomplete",
            "consented": bool(participants),
            "live_browser": all_sessions_live_browser,
            "physical_gpu_12_to_24_gb": all_sessions_physical,
            "client_playback_acknowledgements": acknowledgements_complete,
            "official_e2e_eligible": official_e2e_eligible,
            "official_e2e_rows": official_e2e_rows,
            "official_interrupt_rows": official_interrupt_rows,
            "official_failures_or_timeouts": (
                missing_client_start_ack + missing_client_stop_ack + runtime_failure_rows
            ),
            "official_runtime_failure_rows": runtime_failure_rows,
            "missing_client_playback_started_ack": missing_client_start_ack,
            "missing_client_playback_stopped_ack": missing_client_stop_ack,
            "system_provenance": (
                dict(
                    zip(
                        (
                            "system_name",
                            "asr_backend",
                            "responder_model",
                            "responder_revision",
                            "responder_prompt_profile",
                            "tts_model_sha256",
                        ),
                        next(iter(system_identities)),
                        strict=True,
                    )
                )
                if len(system_identities) == 1
                else None
            ),
            "participants": len(participants),
            "elderly_participants": len(elderly),
            "turns": turns_n,
            "ratings": len(ratings),
            "complete_ratings": len(complete_ratings),
            "invalid_turn_rows": invalid_turn_rows,
            "invalid_rating_rows": invalid_rating_rows,
            "rating_means": rating_means,
            "rating_mean_ci95": rating_ci95,
            "mos_mean": naturalness_mos,
            "mos_mean_ci95": rating_ci95.get("naturalness"),
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
            "official_t_first_audio_max_ms": official_first_audio_max,
            "official_t_barge_in_p95_ms": official_barge_in_p95,
            "official_t_barge_in_max_ms": official_barge_in_max,
            "official_latency_gate_passed": official_latency_gate_passed,
            "official_barge_in_engineering_gate_passed": (
                official_barge_in_engineering_gate_passed
            ),
            "official_interrupt_latency_le_150_ms": official_interrupt_latency_le_150_ms,
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
            write_json(Path(out_json), report)
        return report
