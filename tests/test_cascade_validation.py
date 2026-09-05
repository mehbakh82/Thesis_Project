from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from scripts.evaluate_cascade_validation import USER_CHANNEL_INDEX, read_user_channel


def test_validation_reader_selects_audited_user_channel_one(tmp_path: Path) -> None:
    path = tmp_path / "pair.wav"
    frames = np.asarray([[100, 1000], [-100, -1000]], dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(frames.tobytes())

    audio, metadata = read_user_channel(path)

    assert USER_CHANNEL_INDEX == 1
    assert metadata["selected_user_channel"] == 1
    assert np.allclose(audio, np.asarray([1000, -1000]) / 32768.0)
