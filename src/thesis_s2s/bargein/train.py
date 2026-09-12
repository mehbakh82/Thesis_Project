"""Train the GBDT barge-in detector on synthetic (and recorded, when present) duplex clips."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from thesis_s2s.audio import read_wav
from thesis_s2s.bargein.detector import (
    LABELS,
    BargeinDetector,
    DetectorConfig,
    EnergyVadBaseline,
    evaluate_detectors,
)
from thesis_s2s.bargein.features import FeatureConfig
from thesis_s2s.bargein.synthetic import DuplexClip, make_dataset
from thesis_s2s.config import portable_path, project_root
from thesis_s2s.metrics import binary_score_confidence_intervals, binary_scores, write_json


def _hop_cpu_ms(detector: BargeinDetector, audio) -> float:
    tail = audio[: int(0.45 * 16000)]
    t0 = time.perf_counter()
    detector.interrupt_now(tail)
    return 1000.0 * (time.perf_counter() - t0)


def clip_from_labeled_wav(
    audio: np.ndarray, kind: str, feat_cfg: FeatureConfig | None = None, group_id: str | None = None
) -> DuplexClip:
    feat_cfg = feat_cfg or FeatureConfig()
    if kind not in LABELS:
        raise ValueError(f"unknown interruption label: {kind!r}")
    n = max(1, 1 + (len(audio) - feat_cfg.win) // feat_cfg.hop)
    labels: np.ndarray = np.zeros(n, dtype=np.int32)
    if kind == "interrupt":
        labels[n // 3 :] = LABELS.index("interrupt")
    elif kind == "backchannel":
        mid = max(0, n // 2)
        labels[mid : mid + 3] = LABELS.index("backchannel")
    elif kind == "noise":
        labels[:] = LABELS.index("noise")
    return DuplexClip(
        audio=np.asarray(audio, dtype=np.float32),
        frame_labels=labels,
        binary_interrupt=int(kind == "interrupt"),
        assistant_mask=np.ones(len(audio), dtype=np.float32),
        kind=kind,
        group_id=group_id,
    )


def clips_from_jsonl(jsonl: Path, *, max_clips: int | None = None) -> list[DuplexClip]:
    clips: list[DuplexClip] = []
    if max_clips is not None and max_clips < 0:
        raise ValueError("max_clips must be non-negative")
    if max_clips == 0:
        return clips
    if not jsonl.is_file():
        return clips
    for line_number, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {jsonl}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {jsonl}:{line_number}")
        path = (
            row.get("interaction_audio_filepath")
            or row.get("audio_filepath")
            or row.get("audio_path")
        )
        if not path:
            continue
        if not Path(path).is_file():
            raise FileNotFoundError(f"recorded detector audio missing at {jsonl}:{line_number}: {path}")
        kind = str(row.get("interrupt_label") or "none")
        if kind not in LABELS:
            raise ValueError(f"unknown interruption label at {jsonl}:{line_number}: {kind!r}")
        try:
            audio, _ = read_wav(path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to read recorded detector audio at {jsonl}:{line_number}: {path}"
            ) from exc
        group_id = str(row.get("speaker_id") or row.get("session_id") or "").strip()
        if not group_id:
            raise ValueError(
                f"recorded detector row lacks speaker/session group at {jsonl}:{line_number}"
            )
        clips.append(clip_from_labeled_wav(audio, kind, group_id=group_id))
        if max_clips is not None and len(clips) >= max_clips:
            break
    return clips


def feature_rows_from_jsonl(jsonl: Path) -> list[dict]:
    """Load lossy aggregate feature rows; raw waveforms are never required."""

    rows: list[dict] = []
    if not jsonl.is_file():
        return rows
    for line_number, line in enumerate(jsonl.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {jsonl}:{line_number}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object at {jsonl}:{line_number}")
        vector = np.asarray(row.get("privacy_feature_vector") or [], dtype=np.float32)
        if vector.size == 0:
            continue
        kind = str(row.get("interrupt_label") or "none")
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"invalid privacy feature vector at {jsonl}:{line_number}")
        if kind not in LABELS:
            raise ValueError(f"unknown interruption label at {jsonl}:{line_number}: {kind!r}")
        group = str(row.get("speaker_id") or row.get("session_id") or "").strip()
        if not group:
            raise ValueError(f"feature row lacks speaker/session group at {jsonl}:{line_number}")
        rows.append(
            {
                "vector": vector,
                "label": LABELS.index(kind),
                "binary": int(kind == "interrupt"),
                "group": group,
                "kind": kind,
            }
        )
    dimensions = {len(row["vector"]) for row in rows}
    if len(dimensions) > 1:
        raise ValueError(f"privacy feature dimensions are inconsistent: {sorted(dimensions)}")
    return rows


def train_feature_detector(
    rows: list[dict],
    *,
    seed: int,
    out_dir: Path,
    detector_config: DetectorConfig | None = None,
) -> dict:
    """Train/evaluate by speaker on aggregate features with no retained voice."""

    groups = sorted({str(row["group"]) for row in rows})
    if len(groups) < 2:
        raise ValueError("feature-only held-out evaluation requires at least two speakers/sessions")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test_groups = max(1, round(0.2 * len(groups)))
    test_groups = set(groups[-n_test_groups:])
    train_rows = [row for row in rows if row["group"] not in test_groups]
    test_rows = [row for row in rows if row["group"] in test_groups]
    y_train_binary = {int(row["binary"]) for row in train_rows}
    y_test_binary = {int(row["binary"]) for row in test_rows}
    if y_train_binary != {0, 1} or y_test_binary != {0, 1}:
        raise ValueError(
            "every held-out fold needs interrupt and non-interrupt examples; "
            "run all scripted prompts for every participant"
        )
    x_train = np.stack([row["vector"] for row in train_rows])
    y_train = np.asarray([row["label"] for row in train_rows], dtype=int)
    x_test = np.stack([row["vector"] for row in test_rows])
    y_true = [int(row["binary"]) for row in test_rows]
    detector = BargeinDetector(det=detector_config)
    detector.fit_vectors(x_train, y_train)
    predicted_labels = detector.predict_vectors(x_test)
    y_pred = [int(value == LABELS.index("interrupt")) for value in predicted_labels]
    proposed = binary_scores(y_true, y_pred)

    # Aggregate vectors are mean/std/max blocks. The first two mean features
    # are log-energy and ZCR, so the classical baseline remains reproducible.
    baseline_pred = [int(row[0] > -35.0 and row[1] < 0.25) for row in x_test]
    baseline = binary_scores(y_true, baseline_pred)
    train_groups = {str(row["group"]) for row in train_rows}
    report = {
        "evidence_class": "recorded_features_heldout",
        "label_source": "session_event_metadata_not_independent_human_review",
        "human_verified_labels": 0,
        "official_detector_eligible": False,
        "official_target_satisfied": False,
        "privacy_scope": "lossy aggregate features; no waveform retained",
        "feature_schema": "energy-zcr-f0-voicing-mfcc13-delta13.context400.mean-std-max.v1",
        "n": len(test_rows),
        "n_train": len(train_rows),
        "n_groups": len(groups),
        "test_groups": sorted(test_groups),
        "group_overlap": bool(train_groups & test_groups),
        "split_unit": "speaker_or_session",
        "proposed": proposed.__dict__,
        "proposed_ci95": binary_score_confidence_intervals(y_true, y_pred),
        "energy_vad_baseline": baseline.__dict__,
        "energy_vad_baseline_ci95": binary_score_confidence_intervals(y_true, baseline_pred),
        "classification_report_proposed": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["other", "interrupt"],
            zero_division=0,
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "bargein_gbdt.pkl"
    detector.save(model_path)
    report["model_path"] = portable_path(model_path, root=project_root())
    write_json(out_dir / "feature_heldout_report.json", report)
    return report


def default_recorded_jsonl() -> Path:
    return project_root() / "data" / "processed" / "manifests" / "recorded.jsonl"


def train_and_eval(
    n_per_class: int = 36,
    seed: int = 0,
    out_dir: str | Path | None = None,
    recorded_jsonl: Path | None = None,
    detector_config: DetectorConfig | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    synth = make_dataset(n_per_class=n_per_class, seed=seed)
    recorded_jsonl = (
        Path(recorded_jsonl) if recorded_jsonl is not None else default_recorded_jsonl()
    )
    resolved_out_dir = Path(out_dir or project_root() / "results" / "bargein")
    feature_rows = feature_rows_from_jsonl(recorded_jsonl)
    if feature_rows:
        return train_feature_detector(
            feature_rows, seed=seed, out_dir=resolved_out_dir, detector_config=detector_config
        )
    recorded = clips_from_jsonl(recorded_jsonl)
    split_s = int(0.8 * len(synth))
    train_s, test_s = synth[:split_s], synth[split_s:]
    if recorded:
        groups = sorted({str(clip.group_id or "unknown") for clip in recorded})
        rng.shuffle(groups)
        n_test_groups = max(1, round(0.2 * len(groups))) if len(groups) >= 2 else 0
        test_groups = set(groups[-n_test_groups:]) if n_test_groups else set()
        train_r = [clip for clip in recorded if clip.group_id not in test_groups]
        test_r = [clip for clip in recorded if clip.group_id in test_groups]
    else:
        train_r, test_r = [], []
    train = train_s + train_r
    detector = BargeinDetector(det=detector_config)
    detector.fit([c.audio for c in train], [c.frame_labels for c in train])
    baseline = EnergyVadBaseline()
    out_dir = resolved_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "bargein_gbdt.pkl"
    detector.save(model_path)
    report = evaluate_detectors(
        [c.audio for c in test_s],
        [c.binary_interrupt for c in test_s],
        proposed=detector,
        baseline=baseline,
        assistant_masks=[c.assistant_mask for c in test_s],
        out_json=out_dir / "heldout_report.json",
    )
    hops = [_hop_cpu_ms(detector, c.audio) for c in test_s[:8]]
    report["evidence_class"] = "synthetic_proxy"
    report["human_verified_labels"] = 0
    report["official_detector_eligible"] = False
    report["official_target_satisfied"] = False
    report["model_path"] = portable_path(model_path, root=project_root())
    report["n_train"] = len(train)
    report["n_test_synthetic"] = len(test_s)
    report["n_test"] = len(test_s)
    report["n_recorded"] = len(recorded)
    report["recorded_split_unit"] = "speaker_or_session"
    report["recorded_group_overlap"] = bool(
        {c.group_id for c in train_r} & {c.group_id for c in test_r}
    )
    report["hop_cpu_ms"] = round(sum(hops) / max(1, len(hops)), 3)
    report["hop_budget_ok"] = report["hop_cpu_ms"] < 20.0
    report["recorded_eval"] = False
    if test_r:
        rec_report = evaluate_detectors(
            [c.audio for c in test_r],
            [c.binary_interrupt for c in test_r],
            proposed=detector,
            baseline=baseline,
            assistant_masks=[c.assistant_mask for c in test_r],
            out_json=out_dir / "recorded_heldout_report.json",
        )
        rec_report["evidence_class"] = "recorded_audio_heldout"
        rec_report["label_source"] = "manifest_event_metadata_not_independent_human_review"
        rec_report["human_verified_labels"] = 0
        rec_report["official_detector_eligible"] = False
        rec_report["official_target_satisfied"] = False
        rec_report["n"] = len(test_r)
        rec_report["group_overlap"] = report["recorded_group_overlap"]
        rec_report["split_unit"] = "speaker_or_session"
        write_json(out_dir / "recorded_heldout_report.json", rec_report)
        rec_prop = rec_report.get("proposed") or {}
        report["recorded_eval"] = True
        report["recorded_heldout"] = rec_prop
        report["recorded_target_ok"] = bool(rec_prop.get("target_ok"))
        report["note"] = (
            "Thesis table uses real speaker/session-held-out evidence, not synthetic 1.00."
        )
    else:
        report["note"] = (
            "No real audio or privacy-feature held-out fold is available; synthetic evidence only."
        )
    write_json(out_dir / "heldout_report.json", report)
    return report
