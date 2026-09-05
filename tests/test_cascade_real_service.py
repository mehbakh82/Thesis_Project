import wave
from pathlib import Path

import numpy as np

from scripts.evaluate_cascade_real_service import read_user_channel, script_statistics


def test_script_statistics_requires_persian_letters() -> None:
    assert script_statistics("سلام، خوبی؟")["persian_letter_fraction"] == 1.0
    assert script_statistics("hello 123")["persian_letter_fraction"] == 0.0


def test_read_user_channel_preserves_channel_zero(tmp_path: Path) -> None:
    path = tmp_path / "pair.wav"
    left = np.asarray([1000, -2000, 3000], dtype="<i2")
    right = np.asarray([-3000, 2000, -1000], dtype="<i2")
    stereo = np.column_stack((left, right)).reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(stereo.tobytes())

    user, metadata = read_user_channel(path)

    assert np.allclose(user, left.astype(np.float32) / 32768.0)
    assert metadata == {
        "sample_rate": 16000,
        "channels": 2,
        "sample_width_bytes": 2,
        "frames": 3,
    }
