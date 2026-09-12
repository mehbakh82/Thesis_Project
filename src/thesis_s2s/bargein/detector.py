"""Energy-VAD baseline and GBDT barge-in detector."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from thesis_s2s.bargein.features import (
    FeatureConfig,
    frame_feature_matrix,
    streaming_context_vectors,
)
from thesis_s2s.metrics import binary_score_confidence_intervals, binary_scores, write_json

LABELS = ("none", "interrupt", "backchannel", "noise")
POSITIVE = "interrupt"
MODEL_SCHEMA_VERSION = 1


@dataclass
class DetectorConfig:
    energy_vad_db: float = -35.0
    energy_vad_zcr_max: float = 0.25
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.08
    min_samples_leaf: int = 8


class EnergyVadBaseline:
    """Simple energy + ZCR detector. Expected high FAR on noise/backchannel."""

    def __init__(self, cfg: FeatureConfig | None = None, det: DetectorConfig | None = None):
        self.feat_cfg = cfg or FeatureConfig()
        self.det = det or DetectorConfig()

    def predict_frames(
        self, audio: np.ndarray, assistant_mask: np.ndarray | None = None
    ) -> np.ndarray:
        samples = _validated_audio(audio)
        if samples.size == 0:
            return np.zeros(0, dtype=np.int32)
        feats = frame_feature_matrix(samples, self.feat_cfg)
        energy = feats[:, 0]
        zcr = feats[:, 1]
        pred = ((energy > self.det.energy_vad_db) & (zcr < self.det.energy_vad_zcr_max)).astype(
            np.int32
        )
        if assistant_mask is not None:
            mask_samples = np.asarray(assistant_mask, dtype=np.float32)
            if (
                mask_samples.ndim != 1
                or mask_samples.size == 0
                or not np.isfinite(mask_samples).all()
            ):
                raise ValueError("assistant_mask must be a non-empty finite one-dimensional array")
            hop = self.feat_cfg.hop
            mask = np.array(
                [
                    mask_samples[min(len(mask_samples) - 1, i * hop)] > 0.5
                    for i in range(len(pred))
                ]
            )
            pred = pred * mask.astype(np.int32)
        return pred

    def predict_binary(self, audio: np.ndarray, assistant_mask: np.ndarray | None = None) -> int:
        frames = self.predict_frames(audio, assistant_mask)
        return int(frames.max()) if len(frames) else 0


def _validated_audio(audio: np.ndarray) -> np.ndarray:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("audio must be a finite one-dimensional array")
    return samples


class BargeinDetector:
    """GBDT on energy / F0 / MFCC context windows. Owns duplex stop-playback."""

    def __init__(self, feat_cfg: FeatureConfig | None = None, det: DetectorConfig | None = None):
        self.feat_cfg = feat_cfg or FeatureConfig()
        self.det = det or DetectorConfig()
        self.pipeline = self._new_pipeline()
        self.fitted = False

    def _new_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "gbdt",
                    GradientBoostingClassifier(
                        n_estimators=self.det.n_estimators,
                        max_depth=self.det.max_depth,
                        learning_rate=self.det.learning_rate,
                        min_samples_leaf=self.det.min_samples_leaf,
                        random_state=0,
                    ),
                ),
            ]
        )

    @property
    def vector_width(self) -> int:
        return 3 * (4 + 2 * self.feat_cfg.n_mfcc)

    def _vectors(self, audio: np.ndarray) -> np.ndarray:
        samples = _validated_audio(audio)
        if samples.size == 0:
            return np.zeros((0, self.vector_width), dtype=np.float32)
        feats = frame_feature_matrix(samples, self.feat_cfg)
        return streaming_context_vectors(feats, self.feat_cfg.context_frames)

    def fit_vectors(self, vectors: np.ndarray, labels: np.ndarray) -> BargeinDetector:
        """Fit already-extracted privacy-reduced context vectors."""

        x = np.asarray(vectors, dtype=np.float32)
        raw_y = np.asarray(labels)
        if (
            x.ndim != 2
            or x.shape[1] != self.vector_width
            or raw_y.ndim != 1
            or len(x) != len(raw_y)
            or len(x) == 0
        ):
            raise ValueError(
                f"vectors must be a non-empty 2-D array with {self.vector_width} columns "
                "aligned with one-dimensional labels"
            )
        try:
            numeric_y = raw_y.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("labels must use the integer detector schema") from exc
        if (
            not np.isfinite(x).all()
            or not np.isfinite(numeric_y).all()
            or not np.equal(numeric_y, np.floor(numeric_y)).all()
        ):
            raise ValueError(
                "feature vectors must be finite and labels must use the detector schema"
            )
        y = numeric_y.astype(np.int32)
        if not np.isin(y, range(len(LABELS))).all():
            raise ValueError(
                "feature vectors must be finite and labels must use the detector schema"
            )
        if np.unique(y).size < 2:
            raise ValueError("detector training requires at least two label classes")
        candidate = self._new_pipeline()
        candidate.fit(x, y)
        self.pipeline = candidate
        self.fitted = True
        return self

    def fit(
        self, audios: Sequence[np.ndarray], frame_labels: Sequence[np.ndarray]
    ) -> BargeinDetector:
        if len(audios) != len(frame_labels) or len(audios) == 0:
            raise ValueError("audios and frame_labels must be non-empty aligned sequences")
        xs = []
        ys = []
        for index, (audio, labels) in enumerate(zip(audios, frame_labels, strict=True)):
            vec = self._vectors(audio)
            label_array = np.asarray(labels)
            if vec.shape[0] == 0 or label_array.ndim != 1 or len(vec) != len(label_array):
                raise ValueError(
                    f"audio {index} must be non-empty and have one label per feature frame"
                )
            xs.append(vec)
            ys.append(label_array)
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return self.fit_vectors(x, y)

    def predict_frames(self, audio: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("detector is not fitted")
        vec = self._vectors(audio)
        if len(vec) == 0:
            return np.zeros(0, dtype=np.int32)
        return self.pipeline.predict(vec).astype(np.int32)

    def predict_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("detector is not fitted")
        x = np.asarray(vectors, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.vector_width or not np.isfinite(x).all():
            raise ValueError(f"vectors must be a finite 2-D array with {self.vector_width} columns")
        if len(x) == 0:
            return np.zeros(0, dtype=np.int32)
        return self.pipeline.predict(x).astype(np.int32)

    def predict_binary(self, audio: np.ndarray) -> int:
        frames = self.predict_frames(audio)
        return int((frames == LABELS.index(POSITIVE)).any()) if len(frames) else 0

    def interrupt_now(self, audio_tail: np.ndarray) -> bool:
        """Realtime API: True means stop playback and drop remaining tokens."""

        return bool(self.predict_binary(audio_tail))

    def save(self, path: str | Path) -> None:
        """Atomically save a fitted detector for later trusted-local loading."""

        if not self.fitted:
            raise RuntimeError("cannot save an unfitted detector")
        check_is_fitted(self.pipeline)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                pickle.dump(
                    {
                        "schema_version": MODEL_SCHEMA_VERSION,
                        "feat_cfg": self.feat_cfg,
                        "det": self.det,
                        "pipeline": self.pipeline,
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> BargeinDetector:
        """Load a project-created detector. Pickle files from untrusted sources are unsafe."""

        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("detector payload must be a mapping")
        schema_version = payload.get("schema_version", 0)
        if schema_version not in {0, MODEL_SCHEMA_VERSION}:
            raise ValueError(f"unsupported detector schema version: {schema_version!r}")
        if not {"feat_cfg", "det", "pipeline"}.issubset(payload):
            raise ValueError("detector payload is missing required fields")
        feat_cfg = payload["feat_cfg"]
        det = payload["det"]
        pipeline = payload["pipeline"]
        if (
            not isinstance(feat_cfg, FeatureConfig)
            or not isinstance(det, DetectorConfig)
            or not isinstance(pipeline, Pipeline)
            or not isinstance(pipeline.named_steps.get("scale"), StandardScaler)
            or not isinstance(pipeline.named_steps.get("gbdt"), GradientBoostingClassifier)
        ):
            raise ValueError("detector payload has incompatible component types")
        check_is_fitted(pipeline)
        obj = cls(feat_cfg, det)
        if int(pipeline.n_features_in_) != obj.vector_width:
            raise ValueError("detector payload feature width does not match its feature config")
        obj.pipeline = pipeline
        obj.fitted = True
        return obj


def evaluate_detectors(
    audios: Sequence[np.ndarray],
    binary_labels: Sequence[int],
    *,
    proposed: BargeinDetector,
    baseline: EnergyVadBaseline,
    assistant_masks: Sequence[np.ndarray] | None = None,
    out_json: str | Path | None = None,
) -> dict:
    if len(audios) == 0 or len(audios) != len(binary_labels):
        raise ValueError("evaluation audio and labels must be non-empty and aligned")
    if not set(binary_labels).issubset({0, 1}):
        raise ValueError("evaluation labels must be binary")
    if assistant_masks is not None and len(assistant_masks) != len(audios):
        raise ValueError("assistant masks must align with evaluation audio")
    y = list(binary_labels)
    pred_p = [proposed.predict_binary(a) for a in audios]
    masks: Sequence[np.ndarray | None] = (
        assistant_masks if assistant_masks is not None else [None] * len(audios)
    )
    pred_b = [baseline.predict_binary(a, m) for a, m in zip(audios, masks, strict=True)]
    proposed_scores = binary_scores(y, pred_p)
    baseline_scores = binary_scores(y, pred_b)
    report = {
        "proposed": proposed_scores.__dict__,
        "proposed_ci95": binary_score_confidence_intervals(y, pred_p),
        "energy_vad_baseline": baseline_scores.__dict__,
        "energy_vad_baseline_ci95": binary_score_confidence_intervals(y, pred_b),
        "classification_report_proposed": classification_report(
            y,
            pred_p,
            labels=[0, 1],
            target_names=["other", "interrupt"],
            zero_division=0,
        ),
    }
    if out_json:
        write_json(out_json, report)
    return report
