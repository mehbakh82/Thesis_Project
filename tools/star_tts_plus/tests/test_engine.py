import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from star_tts_plus.engine import SOURCE_RATE, resample, split_punct, split_service  # noqa: E402


@pytest.mark.parametrize("seconds", [0.05, 0.7, 3.3])
def test_kaiser_best_poly_matches_librosa(seconds: float) -> None:
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * SOURCE_RATE)) / SOURCE_RATE
    wav = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * rng.standard_normal(t.size)).astype(np.float32)
    reference = resample(wav, "kaiser_best")
    fast = resample(wav, "kaiser_best_poly")
    assert fast.shape == reference.shape
    snr = 10 * np.log10(np.sum(reference**2) / np.sum((reference - fast) ** 2))
    assert snr > 90


def test_split_service_matches_lab_rules() -> None:
    text = "سلام. حال شما خوب است، امروز هوا خیلی خوب و آفتابی است و می‌توانیم کمی بیرون برویم و قدم بزنیم"
    chunks = split_service(text)
    assert all(len(c.split()) <= 10 for c in chunks)
    assert " ".join(chunks).split() == text.replace(".", " ").replace("،", " ").split()


def test_split_punct_keeps_punctuation_and_caps_length() -> None:
    text = "آیا دارو را خوردید؟ " + " ".join(["کلمه"] * 30) + ". بله!"
    chunks = split_punct(text)
    assert chunks[0] == "آیا دارو را خوردید؟"
    assert all(len(c.split()) <= 25 for c in chunks)
    assert chunks[-1].endswith("بله!")  # a short fragment merges into the previous chunk
