"""Session-held-out recorded-audio proxy benchmark for interruption detection."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import classification_report

from thesis_s2s.audio import read_wav
from thesis_s2s.bargein.detector import BargeinDetector, DetectorConfig
from thesis_s2s.bargein.features import FeatureConfig, context_vector, frame_feature_matrix
from thesis_s2s.data.conversation import (
    SpeakerSegment,
    _automatic_interaction_candidate,
    _merge_same_speaker,
    _segments,
)
from thesis_s2s.data.interaction_qa import _window_candidates
from thesis_s2s.metrics import (
    binary_score_confidence_intervals,
    binary_scores,
    write_json,
)


@dataclass(frozen=True)
class RecordedEvent:
    event_id: str
    session_id: str
    channel: str
    audio_path: str
    onset_s: float
    label: int
    kind: str
    split: str
    source_window_id: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(session_id: str, *, seed: int, train_end: int, validation_end: int) -> str:
    digest = hashlib.sha256(f"{seed}:{session_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _raw_turns(row: dict[str, Any]) -> list[tuple[float, float, str]]:
    turns: list[tuple[float, float, str]] = []
    for item in row.get("speaker_turns") or []:
        if not isinstance(item, dict):
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
        if speaker and 0 <= start < end:
            turns.append((start, end, speaker))
    return turns


def _match_ratio(segment: SpeakerSegment, start: float, end: float) -> float:
    denominator = min(segment.duration, end - start)
    return (
        _overlap(segment.start, segment.end, start, end) / denominator if denominator > 0 else 0.0
    )


def _clean_turn_onset(
    row: dict[str, Any],
    user: SpeakerSegment,
    response: SpeakerSegment,
    *,
    match_ratio: float,
    boundary_tolerance_s: float,
    minimum_gap_s: float,
    maximum_gap_s: float,
) -> float | None:
    user_turns = [
        (start, end, _match_ratio(user, start, end))
        for start, end, speaker in _raw_turns(row)
        if speaker == user.speaker
        and _match_ratio(user, start, end) >= match_ratio
        and abs(end - user.end) <= boundary_tolerance_s
    ]
    response_turns = [
        (start, end, _match_ratio(response, start, end))
        for start, end, speaker in _raw_turns(row)
        if speaker == response.speaker
        and _match_ratio(response, start, end) >= match_ratio
        and abs(start - response.start) <= boundary_tolerance_s
    ]
    candidates: list[tuple[float, float, float]] = []
    for _user_start, user_end, user_match in user_turns:
        for response_start, _response_end, response_match in response_turns:
            gap = response_start - user_end
            if minimum_gap_s <= gap <= maximum_gap_s:
                candidates.append((min(user_match, response_match), -gap, response_start))
    if not candidates:
        return None
    return max(candidates)[2]


def _eligible_row(row: dict[str, Any]) -> bool:
    return (
        row.get("automatic_multi_speaker_verified") is True
        and row.get("reference_alignment_status") == "complete"
        and row.get("internal_research_authorized") is True
        and not (row.get("manual_qa_reviewed") is True and row.get("human_verified") is not True)
    )


def discover_events(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[RecordedEvent]:
    event_cfg = config["events"]
    split_cfg = config["split"]
    seed = int(config["seed"])
    events: list[RecordedEvent] = []
    for row in rows:
        if not _eligible_row(row):
            continue
        session_id = str(row.get("session_id") or row.get("episode_id") or "").strip()
        audio_path = str(row.get("audio_filepath") or row.get("audio_path") or "").strip()
        window_id = str(row.get("window_id") or "").strip()
        channel = str(row.get("channel") or "unknown")
        if not session_id or not audio_path or not window_id:
            continue
        split = stable_split(
            session_id,
            seed=seed,
            train_end=int(split_cfg["train_end_exclusive"]),
            validation_end=int(split_cfg["validation_end_exclusive"]),
        )
        for candidate in _window_candidates(row):
            kind = str(candidate["automatic_candidate"])
            if kind not in {"interrupt", "backchannel"}:
                continue
            events.append(
                RecordedEvent(
                    event_id=str(candidate["candidate_id"]),
                    session_id=session_id,
                    channel=channel,
                    audio_path=audio_path,
                    onset_s=float(candidate["raw_response_start_s"]),
                    label=int(kind == "interrupt"),
                    kind=kind,
                    split=split,
                    source_window_id=window_id,
                )
            )

        segments = _merge_same_speaker(_segments(row))
        index = 0
        while index + 1 < len(segments):
            user, response = segments[index], segments[index + 1]
            index += 2
            if (
                user.speaker == response.speaker
                or response.start - user.end > float(event_cfg["maximum_clean_gap_s"])
                or user.duration < 0.25
                or response.duration < 0.25
                or _automatic_interaction_candidate(row, user, response) is not None
            ):
                continue
            onset = _clean_turn_onset(
                row,
                user,
                response,
                match_ratio=float(event_cfg["raw_turn_match_ratio"]),
                boundary_tolerance_s=float(event_cfg["boundary_tolerance_s"]),
                minimum_gap_s=float(event_cfg["minimum_clean_gap_s"]),
                maximum_gap_s=float(event_cfg["maximum_clean_gap_s"]),
            )
            if onset is None:
                continue
            identity = (
                f"{window_id}|{user.speaker}|{response.speaker}|"
                f"{user.start:.3f}|{response.start:.3f}|clean_turn"
            )
            events.append(
                RecordedEvent(
                    event_id="clean-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
                    session_id=session_id,
                    channel=channel,
                    audio_path=audio_path,
                    onset_s=onset,
                    label=0,
                    kind="clean_turn",
                    split=split,
                    source_window_id=window_id,
                )
            )
    identifiers = [event.event_id for event in events]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("recorded proxy event identifiers are not unique")
    return events


def _stable_events(events: list[RecordedEvent]) -> list[RecordedEvent]:
    return sorted(
        events,
        key=lambda event: hashlib.sha256(event.event_id.encode()).hexdigest(),
    )


def select_balanced_events(
    events: list[RecordedEvent], *, maximum_events_per_kind_per_session: int
) -> list[RecordedEvent]:
    capped: list[RecordedEvent] = []
    by_session_kind: dict[tuple[str, str, str], list[RecordedEvent]] = defaultdict(list)
    for event in events:
        by_session_kind[(event.split, event.session_id, event.kind)].append(event)
    for key in sorted(by_session_kind):
        capped.extend(_stable_events(by_session_kind[key])[:maximum_events_per_kind_per_session])

    selected: list[RecordedEvent] = []
    for split in ("train", "validation", "test"):
        positives = _stable_events(
            [event for event in capped if event.split == split and event.label == 1]
        )
        backchannels = _stable_events(
            [event for event in capped if event.split == split and event.kind == "backchannel"]
        )
        clean_turns = _stable_events(
            [event for event in capped if event.split == split and event.kind == "clean_turn"]
        )
        negatives = backchannels[: len(positives)]
        if len(negatives) < len(positives):
            negatives.extend(clean_turns[: len(positives) - len(negatives)])
        selected.extend(positives)
        selected.extend(negatives)
    return sorted(selected, key=lambda event: (event.split, event.event_id))


def validate_event_split(events: list[RecordedEvent], config: dict[str, Any]) -> dict[str, bool]:
    split_cfg = config["split"]
    groups = {
        split: {event.session_id for event in events if event.split == split}
        for split in ("train", "validation", "test")
    }
    counts = Counter((event.split, event.label) for event in events)
    minimum_groups = int(split_cfg["minimum_groups_per_split"])
    minimum_class = int(split_cfg["minimum_events_per_class_per_split"])
    return {
        "all_three_splits_present": all(
            any(event.split == split for event in events) for split in groups
        ),
        "session_groups_disjoint": not (
            groups["train"] & groups["validation"]
            or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"]
        ),
        "minimum_groups_per_split": all(len(value) >= minimum_groups for value in groups.values()),
        "both_classes_and_minimum_events_per_split": all(
            counts[(split, label)] >= minimum_class for split in groups for label in (0, 1)
        ),
        "balanced_classes_per_split": all(
            counts[(split, 0)] == counts[(split, 1)] for split in groups
        ),
    }


def extract_feature_rows(
    events: list[RecordedEvent],
    *,
    root: Path,
    feature_config: FeatureConfig,
    pre_onset_s: float,
    post_onset_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[int] = []
    by_audio: dict[str, list[RecordedEvent]] = defaultdict(list)
    for event in events:
        by_audio[event.audio_path].append(event)
    total_samples = int(round((pre_onset_s + post_onset_s) * feature_config.sample_rate))
    for raw_path in sorted(by_audio):
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"recorded proxy source audio missing: {path}")
        audio, sample_rate = read_wav(path, target_sr=feature_config.sample_rate)
        if sample_rate != feature_config.sample_rate:
            raise ValueError(f"unexpected sample rate after resampling: {sample_rate}")
        for event in by_audio[raw_path]:
            start = int(round((event.onset_s - pre_onset_s) * sample_rate))
            end = start + total_samples
            clip: np.ndarray = np.zeros(total_samples, dtype=np.float32)
            source_start, source_end = max(0, start), min(len(audio), end)
            if source_end > source_start:
                target_start = source_start - start
                clip[target_start : target_start + source_end - source_start] = audio[
                    source_start:source_end
                ]
            features = frame_feature_matrix(clip, feature_config)
            vectors.append(
                context_vector(features, len(features) - 1, feature_config.context_frames)
            )
            labels.append(event.label)
    if len(vectors) != len(events):
        raise RuntimeError("feature extraction did not preserve event cardinality")
    return np.stack(vectors), np.asarray(labels, dtype=np.int32)


def _scores_with_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    scores = binary_scores(y_true.tolist(), y_pred.tolist())
    tn, fp = scores.matrix[0]
    fn, tp = scores.matrix[1]
    tnr = tn / max(1, tn + fp)
    tpr = tp / max(1, tp + fn)
    return {**asdict(scores), "balanced_accuracy": (tnr + tpr) / 2.0}


def select_validation_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    candidates: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in candidates:
        predicted = (probabilities >= threshold).astype(np.int32)
        rows.append(
            {
                "threshold": float(threshold),
                **_scores_with_balanced_accuracy(y_true, predicted),
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["interrupt_f1"]),
            -abs(float(row["threshold"]) - 0.5),
            -float(row["threshold"]),
        ),
    )
    return float(selected["threshold"]), rows


def group_bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[str],
    *,
    seed: int,
    samples: int = 2000,
) -> dict[str, list[float]]:
    group_names = sorted(set(groups))
    indices = {
        group: np.asarray(
            [index for index, value in enumerate(groups) if value == group], dtype=int
        )
        for group in group_names
    }
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        chosen = rng.choice(group_names, size=len(group_names), replace=True)
        selected = np.concatenate([indices[str(group)] for group in chosen])
        score = binary_scores(y_true[selected].tolist(), y_pred[selected].tolist())
        values["accuracy"].append(score.accuracy)
        values["interrupt_f1"].append(score.interrupt_f1)
        values["far"].append(score.far)
        values["frr"].append(score.frr)
    intervals: dict[str, list[float]] = {}
    for name, rows in values.items():
        bounds: np.ndarray = np.asarray(np.percentile(rows, [2.5, 97.5]))
        intervals[name] = [round(float(bound), 4) for bound in bounds]
    return intervals


def _event_counts(events: list[RecordedEvent]) -> dict[str, Any]:
    return {
        "by_split_and_kind": {
            f"{split}/{kind}": count
            for (split, kind), count in sorted(Counter((e.split, e.kind) for e in events).items())
        },
        "sessions_by_split": {
            split: len({event.session_id for event in events if event.split == split})
            for split in ("train", "validation", "test")
        },
        "channels_by_split": {
            split: dict(sorted(Counter(e.channel for e in events if e.split == split).items()))
            for split in ("train", "validation", "test")
        },
    }


def evaluate_recorded_proxy(
    *,
    config_path: Path,
    out_path: Path | None = None,
    root: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("recorded proxy config must be a mapping")
    output_cfg = config["output"]
    report_path = out_path or root / str(output_cfg["report"])
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite recorded proxy report: {report_path}")
    source_path = root / str(config["source_manifest"])
    protocol_path = root / str(config["protocol_document"])
    if sha256_file(source_path) != str(config["source_manifest_sha256"]):
        raise RuntimeError("recorded proxy source manifest hash drifted after protocol freeze")
    rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    discovered = discover_events(rows, config)
    events = select_balanced_events(
        discovered,
        maximum_events_per_kind_per_session=int(
            config["events"]["maximum_events_per_kind_per_session"]
        ),
    )
    requirements = validate_event_split(events, config)
    if not all(requirements.values()):
        raise RuntimeError(f"recorded proxy split failed closed: {requirements}")

    feature_cfg = FeatureConfig(
        sample_rate=int(config["features"]["sample_rate"]),
        frame_ms=float(config["features"]["frame_ms"]),
        context_ms=float(config["features"]["context_ms"]),
    )
    x, y = extract_feature_rows(
        events,
        root=root,
        feature_config=feature_cfg,
        pre_onset_s=float(config["features"]["pre_onset_s"]),
        post_onset_s=float(config["features"]["post_onset_s"]),
    )
    indices = {
        split: np.asarray([i for i, event in enumerate(events) if event.split == split], dtype=int)
        for split in ("train", "validation", "test")
    }
    model_cfg = config["model"]
    detector = BargeinDetector(
        feat_cfg=feature_cfg,
        det=DetectorConfig(
            n_estimators=int(model_cfg["n_estimators"]),
            max_depth=int(model_cfg["max_depth"]),
            learning_rate=float(model_cfg["learning_rate"]),
            min_samples_leaf=int(model_cfg["min_samples_leaf"]),
        ),
    )
    detector.fit_vectors(x[indices["train"]], y[indices["train"]])
    classes = detector.pipeline.named_steps["gbdt"].classes_.tolist()
    positive_column = classes.index(1)
    validation_probability = detector.pipeline.predict_proba(x[indices["validation"]])[
        :, positive_column
    ]
    threshold, threshold_rows = select_validation_threshold(
        y[indices["validation"]],
        validation_probability,
        [float(value) for value in config["threshold"]["candidates"]],
    )
    test_probability = detector.pipeline.predict_proba(x[indices["test"]])[:, positive_column]
    test_prediction = (test_probability >= threshold).astype(np.int32)
    test_truth = y[indices["test"]]
    test_scores = _scores_with_balanced_accuracy(test_truth, test_prediction)
    baseline_prediction = ((x[indices["test"], 0] > -35.0) & (x[indices["test"], 1] < 0.25)).astype(
        np.int32
    )
    baseline_scores = _scores_with_balanced_accuracy(test_truth, baseline_prediction)
    test_groups = [events[index].session_id for index in indices["test"]]

    model_path = root / str(output_cfg["model"])
    detector.save(model_path)
    event_identity = json.dumps(
        [
            {
                "event_id": event.event_id,
                "session_id_hash": hashlib.sha256(event.session_id.encode()).hexdigest(),
                "split": event.split,
                "kind": event.kind,
                "label": event.label,
            }
            for event in events
        ],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "evidence_class": str(config["evidence_class"]),
        "recorded_audio": True,
        "label_source": "automatic_diarization_and_reference_alignment_proxy",
        "human_verified_labels": 0,
        "official_detector_eligible": False,
        "official_ineligibility_reason": (
            "recorded events were not independently human-labeled; automatic proxy "
            "accuracy cannot satisfy the official >80% ground-truth gate"
        ),
        "authorization_scope": "supervisor-approved internal research; no redistribution",
        "raw_audio_retained_by_benchmark": False,
        "feature_vectors_retained": False,
        "group_split_unit": "source_session_id",
        "group_overlap": False,
        "requirements": requirements,
        "events": {
            "discovered": len(discovered),
            "selected": len(events),
            **_event_counts(events),
            "identity_sha256": hashlib.sha256(event_identity).hexdigest(),
        },
        "model": {
            "type": "GradientBoostingClassifier",
            "config": model_cfg,
            "feature_schema": config["features"]["schema"],
            "decision_horizon_after_onset_ms": round(
                1000 * float(config["features"]["post_onset_s"]), 3
            ),
            "path": str(model_path.relative_to(root)),
            "sha256": sha256_file(model_path),
        },
        "threshold_selection": {
            "scope": "validation_sessions_only",
            "criterion": config["threshold"]["criterion"],
            "selected": threshold,
            "candidates": threshold_rows,
        },
        "heldout_test": {
            "evaluated_once": True,
            "n": int(len(test_truth)),
            "sessions": len(set(test_groups)),
            "proposed": test_scores,
            "proposed_event_ci95": binary_score_confidence_intervals(
                test_truth.tolist(), test_prediction.tolist(), seed=int(config["seed"])
            ),
            "proposed_session_block_bootstrap_ci95": group_bootstrap_intervals(
                test_truth,
                test_prediction,
                test_groups,
                seed=int(config["seed"]),
            ),
            "energy_zcr_baseline": baseline_scores,
            "energy_zcr_baseline_event_ci95": binary_score_confidence_intervals(
                test_truth.tolist(), baseline_prediction.tolist(), seed=int(config["seed"])
            ),
            "classification_report": classification_report(
                test_truth,
                test_prediction,
                labels=[0, 1],
                target_names=["other", "interrupt"],
                zero_division=0,
            ),
            "recorded_proxy_accuracy_above_80_percent": (
                float(test_scores["accuracy"]) >= float(config["threshold"]["test_target_accuracy"])
            ),
            "official_target_satisfied": False,
        },
        "artifacts": {
            "source_manifest": str(source_path.relative_to(root)),
            "source_manifest_sha256": sha256_file(source_path),
            "config": str(config_path.relative_to(root)),
            "config_sha256": sha256_file(config_path),
            "protocol": str(protocol_path.relative_to(root)),
            "protocol_sha256": sha256_file(protocol_path),
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
        },
        "interpretation": (
            "This is real recorded-audio, session-held-out proxy evidence. It measures "
            "agreement with frozen automatic diarization labels, not human ground truth."
        ),
    }
    write_json(report_path, report)
    return report
