from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from thesis_s2s.data.romanize import alignment_letters, phonemes_to_words, romanize

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_espeak_persian_phonemes_map_to_ascii_words() -> None:
    sentence = list("salˈɑm, hˈɑl ʃˈomɑ tʃˈetoːr ˈast?")
    assert phonemes_to_words([sentence]) == ["salaam,", "haal", "shomaa", "chetor", "ast?"]


def test_multi_symbol_sequences_and_uvular_marker() -> None:
    words = phonemes_to_words([list("dʒˈɑleb vɑq1ˈeʔan xˈuːb ʒˈɑle")])
    assert words == ["jaaleb", "vaaghean", "khub", "zhaale"]


def test_unmapped_symbol_fails_closed() -> None:
    with pytest.raises(ValueError, match="unmapped phoneme"):
        phonemes_to_words([["s", "ʘ"]])


def test_romanize_uses_the_supplied_phonemizer() -> None:
    class Fake:
        def phonemize(self, text: str) -> list[list[str]]:
            assert text == "سلام"
            return [list("salˈɑm.")]

    assert romanize("سلام", Fake()) == ["salaam."]


def test_alignment_letters_strips_punctuation_and_rejects_empty() -> None:
    assert alignment_letters("ast?") == "ast"
    with pytest.raises(ValueError):
        alignment_letters("?")


@dataclass
class _Alignment:
    phoneme: str
    num_samples: int


@dataclass
class _Chunk:
    sample_rate: int
    phoneme_alignments: list[_Alignment]
    audio_float_array: np.ndarray


def test_word_timings_follow_piper_sample_alignment() -> None:
    prepare = _load_script("prepare_moshi_v7")
    rate = 1000
    alignments = [
        _Alignment("^", 100),
        _Alignment("s", 50),
        _Alignment("ˈ", 10),
        _Alignment("ɑ", 40),
        _Alignment(" ", 200),
        _Alignment("b", 30),
        _Alignment("ɑ", 70),
        _Alignment(",", 20),
        _Alignment("$", 10),
    ]
    total = sum(item.num_samples for item in alignments)
    chunk = _Chunk(rate, alignments, np.zeros(total, dtype=np.float32))
    words, audio, sample_rate = prepare.word_timings([chunk])
    assert sample_rate == rate and audio.size == total
    assert words == [("saa", 0.1, 0.2), ("baa,", 0.4, 0.5)]


def test_v7_text_gates_use_romanized_prefix() -> None:
    evaluator = _load_script("evaluate_moshi_v7_overfit")
    assert evaluator.normalize_roman("Salaam, haal!") == "salaam haal"
    assert evaluator.ascii_letter_fraction("salaam سلام") == pytest.approx(6 / 10)
    assert evaluator.prefix_cer("salaam", "salaam haal shomaa") == 0.0
    assert evaluator.prefix_cer("", "salaam") == 1.0
    assert evaluator.prefix_cer("xxxxxx", "salaam") == 1.0
