import pickle

import numpy as np
import pytest

from thesis_s2s.bargein.detector import (
    BargeinDetector,
    DetectorConfig,
    EnergyVadBaseline,
    evaluate_detectors,
)
from thesis_s2s.bargein.features import FeatureConfig


def _detector() -> BargeinDetector:
    return BargeinDetector(
        det=DetectorConfig(n_estimators=2, max_depth=1, min_samples_leaf=1)
    )


def _training_vectors(detector: BargeinDetector) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.zeros((4, detector.vector_width), dtype=np.float32)
    vectors[2:, 0] = 1.0
    return vectors, np.asarray([0, 0, 1, 1], dtype=np.int32)


def test_vector_fit_requires_schema_width_integer_labels_and_two_classes() -> None:
    detector = _detector()
    vectors, labels = _training_vectors(detector)

    with pytest.raises(ValueError, match="columns"):
        detector.fit_vectors(vectors[:, :-1], labels)
    with pytest.raises(ValueError, match="integer detector schema"):
        detector.fit_vectors(vectors, np.asarray(["none"] * 4))
    with pytest.raises(ValueError, match="at least two"):
        detector.fit_vectors(vectors, np.zeros(4, dtype=np.int32))
    with pytest.raises(ValueError, match="detector schema"):
        detector.fit_vectors(vectors, np.asarray([0, 0, 1.5, 1]))


def test_frame_fit_rejects_empty_or_misaligned_labels() -> None:
    detector = _detector()
    with pytest.raises(ValueError, match="non-empty aligned"):
        detector.fit([], [])
    with pytest.raises(ValueError, match="one label per feature frame"):
        detector.fit([np.zeros(640, dtype=np.float32)], [np.zeros(2, dtype=np.int32)])


def test_empty_audio_is_never_an_interruption() -> None:
    baseline = EnergyVadBaseline()
    assert baseline.predict_frames(np.zeros(0, dtype=np.float32)).size == 0
    assert baseline.predict_binary(np.zeros(0, dtype=np.float32)) == 0

    detector = _detector()
    vectors, labels = _training_vectors(detector)
    detector.fit_vectors(vectors, labels)
    assert detector.predict_frames(np.zeros(0, dtype=np.float32)).size == 0
    assert detector.predict_binary(np.zeros(0, dtype=np.float32)) == 0
    assert detector.predict_vectors(np.zeros((0, detector.vector_width), dtype=np.float32)).size == 0


def test_audio_and_assistant_mask_must_be_finite_one_dimensional() -> None:
    baseline = EnergyVadBaseline()
    with pytest.raises(ValueError, match="audio must be"):
        baseline.predict_binary(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="assistant_mask"):
        baseline.predict_binary(np.zeros(640, dtype=np.float32), np.zeros(0, dtype=np.float32))
    with pytest.raises(ValueError, match="assistant_mask"):
        baseline.predict_binary(
            np.zeros(640, dtype=np.float32), np.asarray([float("nan")], dtype=np.float32)
        )


def test_save_requires_fitted_model_and_round_trip_is_compatible(tmp_path) -> None:
    path = tmp_path / "detector.pkl"
    detector = _detector()
    with pytest.raises(RuntimeError, match="unfitted"):
        detector.save(path)
    assert not path.exists()
    assert detector.fitted is False

    vectors, labels = _training_vectors(detector)
    expected = detector.fit_vectors(vectors, labels).predict_vectors(vectors)
    detector.save(path)
    loaded = BargeinDetector.load(path)
    assert np.array_equal(loaded.predict_vectors(vectors), expected)
    assert list(tmp_path.glob(".detector.pkl.*.tmp")) == []


def test_load_rejects_incompatible_payload_after_deserialization(tmp_path) -> None:
    path = tmp_path / "invalid.pkl"
    with path.open("wb") as handle:
        pickle.dump({"schema_version": 999}, handle)
    with pytest.raises(ValueError, match="unsupported detector schema"):
        BargeinDetector.load(path)

    with path.open("wb") as handle:
        pickle.dump({"schema_version": 1}, handle)
    with pytest.raises(ValueError, match="missing required fields"):
        BargeinDetector.load(path)


def test_evaluation_rejects_empty_misaligned_or_nonbinary_inputs() -> None:
    proposed = _detector()
    vectors, labels = _training_vectors(proposed)
    proposed.fit_vectors(vectors, labels)
    baseline = EnergyVadBaseline()
    audio = np.zeros(640, dtype=np.float32)

    with pytest.raises(ValueError, match="non-empty and aligned"):
        evaluate_detectors([], [], proposed=proposed, baseline=baseline)
    with pytest.raises(ValueError, match="non-empty and aligned"):
        evaluate_detectors([audio], [], proposed=proposed, baseline=baseline)
    with pytest.raises(ValueError, match="binary"):
        evaluate_detectors([audio], [2], proposed=proposed, baseline=baseline)
    with pytest.raises(ValueError, match="assistant masks"):
        evaluate_detectors(
            [audio], [0], proposed=proposed, baseline=baseline, assistant_masks=[]
        )


def test_legacy_trusted_payload_remains_loadable(tmp_path) -> None:
    detector = _detector()
    vectors, labels = _training_vectors(detector)
    detector.fit_vectors(vectors, labels)
    path = tmp_path / "legacy.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "feat_cfg": FeatureConfig(),
                "det": detector.det,
                "pipeline": detector.pipeline,
            },
            handle,
        )

    assert BargeinDetector.load(path).fitted is True
