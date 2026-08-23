"""Energy-VAD baseline and GBDT barge-in detector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from thesis_s2s.bargein.features import (
    FeatureConfig,
    frame_feature_matrix,
    streaming_context_vectors,
)
from thesis_s2s.metrics import binary_score_confidence_intervals, binary_scores, write_json

LABELS = ("none", "interrupt", "backchannel", "noise")
POSITIVE = "interrupt"


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
        feats = frame_feature_matrix(audio, self.feat_cfg)
        energy = feats[:, 0]
        zcr = feats[:, 1]
        pred = ((energy > self.det.energy_vad_db) & (zcr < self.det.energy_vad_zcr_max)).astype(
            np.int32
        )
        if assistant_mask is not None:
            hop = self.feat_cfg.hop
            mask = np.array(
                [
                    assistant_mask[min(len(assistant_mask) - 1, i * hop)] > 0.5
                    for i in range(len(pred))
                ]
            )
            pred = pred * mask.astype(np.int32)
        return pred

    def predict_binary(self, audio: np.ndarray, assistant_mask: np.ndarray | None = None) -> int:
        return int(self.predict_frames(audio, assistant_mask).max() if len(audio) else 0)


class BargeinDetector:
    """GBDT on energy / F0 / MFCC context windows. Owns duplex stop-playback."""

    def __init__(self, feat_cfg: FeatureConfig | None = None, det: DetectorConfig | None = None):
        self.feat_cfg = feat_cfg or FeatureConfig()
        self.det = det or DetectorConfig()
        self.pipeline = Pipeline(
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
        self.fitted = False

    def _vectors(self, audio: np.ndarray) -> np.ndarray:
        feats = frame_feature_matrix(audio, self.feat_cfg)
        return streaming_context_vectors(feats, self.feat_cfg.context_frames)

    def fit_vectors(self, vectors: np.ndarray, labels: np.ndarray) -> BargeinDetector:
        """Fit already-extracted privacy-reduced context vectors."""

        x = np.asarray(vectors, dtype=np.float32)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise ValueError("vectors must be a non-empty 2-D array aligned with labels")
        if not np.isfinite(x).all() or not np.isin(y, range(len(LABELS))).all():
            raise ValueError(
                "feature vectors must be finite and labels must use the detector schema"
            )
        self.pipeline.fit(x, y)
        self.fitted = True
        return self

    def fit(
        self, audios: Sequence[np.ndarray], frame_labels: Sequence[np.ndarray]
    ) -> BargeinDetector:
        xs = []
        ys = []
        for audio, labels in zip(audios, frame_labels, strict=True):
            vec = self._vectors(audio)
            n = min(len(vec), len(labels))
            xs.append(vec[:n])
            ys.append(np.asarray(labels[:n], dtype=int))
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        self.pipeline.fit(x, y)
        self.fitted = True
        return self

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
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("vectors must be a finite 2-D array")
        return self.pipeline.predict(x).astype(np.int32)

    def predict_binary(self, audio: np.ndarray) -> int:
        frames = self.predict_frames(audio)
        return int((frames == LABELS.index(POSITIVE)).any()) if len(frames) else 0

    def interrupt_now(self, audio_tail: np.ndarray) -> bool:
        """Realtime API: True means stop playback and drop remaining tokens."""

        return bool(self.predict_binary(audio_tail))

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {"feat_cfg": self.feat_cfg, "det": self.det, "pipeline": self.pipeline}, handle
            )
        self.fitted = True

    @classmethod
    def load(cls, path: str | Path) -> BargeinDetector:
        import pickle

        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        obj = cls(payload["feat_cfg"], payload["det"])
        obj.pipeline = payload["pipeline"]
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
