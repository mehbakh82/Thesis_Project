import sys
from types import SimpleNamespace

import numpy as np

from thesis_s2s.bakeoff import codecs


def test_mulaw_roundtrip_applies_real_eight_bit_quantization():
    audio = np.linspace(-1.0, 1.0, 4097, dtype=np.float32)

    reconstructed, info = codecs.mulaw_roundtrip(audio)

    assert info["bits_per_sample"] == 8
    assert len(np.unique(reconstructed)) <= 256
    assert not np.array_equal(reconstructed, audio)
    assert np.isfinite(reconstructed).all()
    assert float(np.max(np.abs(reconstructed))) <= 1.0 + 1e-6


def test_snr_is_json_safe_when_reference_energy_is_undefined():
    assert codecs._snr_db(np.array([], dtype=np.float32), np.array([], dtype=np.float32)) is None
    assert codecs._snr_db(np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)) is None
    assert codecs._snr_db(np.ones(8, dtype=np.float32), np.full(8, np.nan)) is None
    assert codecs._snr_db(np.ones(8, dtype=np.float32), np.ones(8, dtype=np.float32)) is not None


def test_mel_roundtrip_preserves_short_input_length():
    reconstructed, info = codecs.mel_griffin_roundtrip(np.ones(31, dtype=np.float32))

    assert len(reconstructed) == 31
    assert info["available"] is True
    assert np.isfinite(reconstructed).all()


def test_codec_cer_uses_canonical_whitespace_and_zwnj_rules():
    assert codecs._cer("سلام دنیا", "سلام\nدنیا") == 0.0
    assert codecs._cer("می\u200cروم", "میروم") == 0.0


def test_whisper_temp_file_is_removed_when_wav_write_fails(tmp_path, monkeypatch):
    temporary = tmp_path / "probe.wav"
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(pipeline=lambda *args, **kwargs: lambda *a, **k: {"text": ""}),
    )

    def stable_mkstemp(*, suffix):
        temporary.touch()
        return codecs.os.open(temporary, codecs.os.O_RDONLY), str(temporary)

    monkeypatch.setattr(codecs.tempfile, "mkstemp", stable_mkstemp)
    monkeypatch.setattr(codecs, "write_wav", lambda *args: (_ for _ in ()).throw(OSError("boom")))

    result = codecs.whisper_intelligibility(np.ones(8), np.ones(8))

    assert result["available"] is False
    assert "boom" in result["error"]
    assert not temporary.exists()
