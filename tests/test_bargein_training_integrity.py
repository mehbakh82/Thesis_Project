import json

import numpy as np
import pytest

from thesis_s2s.audio import write_wav
from thesis_s2s.bargein.train import (
    clip_from_labeled_wav,
    clips_from_jsonl,
    feature_rows_from_jsonl,
)


def test_unknown_clip_label_is_never_converted_to_negative():
    with pytest.raises(ValueError, match="unknown interruption label"):
        clip_from_labeled_wav(np.zeros(1600, dtype=np.float32), "typo")


def test_recorded_audio_loader_requires_valid_label_file_and_group(tmp_path):
    audio = tmp_path / "audio.wav"
    write_wav(audio, np.zeros(1600, dtype=np.float32))
    manifest = tmp_path / "recorded.jsonl"

    manifest.write_text(json.dumps({"audio_path": str(audio), "interrupt_label": "typo"}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown interruption label"):
        clips_from_jsonl(manifest)

    manifest.write_text(json.dumps({"audio_path": str(audio), "interrupt_label": "none"}), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks speaker/session group"):
        clips_from_jsonl(manifest)

    manifest.write_text(
        json.dumps(
            {"audio_path": str(tmp_path / "missing.wav"), "interrupt_label": "none", "session_id": "s"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="audio missing"):
        clips_from_jsonl(manifest)
    assert clips_from_jsonl(manifest, max_clips=0) == []
    with pytest.raises(ValueError, match="non-negative"):
        clips_from_jsonl(manifest, max_clips=-1)


def test_feature_loader_skips_absent_vectors_but_rejects_corrupt_evidence(tmp_path):
    manifest = tmp_path / "features.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"session_id": "no-feature", "privacy_feature_vector": []}),
                json.dumps(
                    {
                        "session_id": "valid",
                        "interrupt_label": "interrupt",
                        "privacy_feature_vector": [1.0, 2.0],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    assert len(feature_rows_from_jsonl(manifest)) == 1

    manifest.write_text(
        json.dumps({"session_id": "s", "interrupt_label": "typo", "privacy_feature_vector": [1]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown interruption label"):
        feature_rows_from_jsonl(manifest)

    manifest.write_text(
        json.dumps({"interrupt_label": "none", "privacy_feature_vector": [1]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="lacks speaker/session group"):
        feature_rows_from_jsonl(manifest)
