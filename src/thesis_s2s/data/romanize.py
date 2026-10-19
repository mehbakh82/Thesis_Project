"""Deterministic ASCII romanization of the Persian phonemes Piper actually spoke.

The Moshi inner-monologue text stream emits at most one text token per 12.5 Hz
frame. Persian script costs about six SentencePiece tokens per word in Moshi's
English tokenizer, which cannot keep pace with speech. Romanizing the exact
espeak-ng phoneme sequence that the Mana Piper voice used to render the
assistant channel keeps the text faithful to the audio while using the
tokenizer's well-trained Latin pieces.

The mapping is many-to-one only where Persian orthography itself is
ambiguous in speech (for example both ق and غ are rendered ``gh``). It never
consults the audio, so it is a pure function of the response text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Protocol

# Ordered: multi-symbol sequences are replaced before single symbols.
_SEQUENCES: tuple[tuple[str, str], ...] = (
    ("dʒ", "j"),
    ("tʃ", "ch"),
    ("q1", "gh"),
)
_SYMBOLS: dict[str, str] = {
    # Persian consonants as espeak-ng renders them.
    "ʃ": "sh",
    "ʒ": "zh",
    "x": "kh",
    "q": "gh",
    "j": "y",
    "ɡ": "g",
    "ʔ": "",
    # Persian vowels: /æ/ a, /ɑ/ aa, /e/ e, /o/ o, /i/ i, /u/ u.
    "a": "a",
    "ɑ": "aa",
    "e": "e",
    "o": "o",
    "i": "i",
    "u": "u",
    # Rare phones from embedded English words.
    "ɪ": "i",
    "ə": "e",
    "ɛ": "e",
    "ɹ": "r",
    "ɒ": "o",
    "ʊ": "u",
    "ʌ": "a",
    "ð": "d",
    "ɔ": "o",
    "ɐ": "a",
    "ŋ": "ng",
    "ɜ": "e",
    "θ": "s",
    "w": "w",
    # Suprasegmentals and diacritics carry no letters.
    "ˈ": "",
    "ˌ": "",
    "ː": "",
    "ʲ": "",
    "̩": "",
    "1": "",
}
_PUNCTUATION = {",": ",", ".": ".", "?": "?", "!": "!", ";": ",", ":": ""}
_ASCII_WORD = re.compile(r"^[a-z]+$")
ROMANIZATION_VERSION = "piper-espeak-fa-ascii-v1"


class Phonemizer(Protocol):
    def phonemize(self, text: str) -> list[list[str]]: ...


def _map_word(phones: str) -> str:
    """Map one word's phoneme string; sequences are matched before symbols.

    Mapped output is emitted directly and never re-mapped, so ``dʒ`` becomes
    ``j`` without the single-symbol rule ``j`` -> ``y`` touching it.
    """

    letters: list[str] = []
    index = 0
    while index < len(phones):
        for source, target in _SEQUENCES:
            if phones.startswith(source, index):
                letters.append(target)
                index += len(source)
                break
        else:
            symbol = phones[index]
            if symbol in _SYMBOLS:
                letters.append(_SYMBOLS[symbol])
            elif "a" <= symbol <= "z":
                letters.append(symbol)
            else:
                raise ValueError(f"unmapped phoneme symbol {symbol!r} in {phones!r}")
            index += 1
    return "".join(letters)


def phonemes_to_words(sentences: Iterable[Sequence[str]]) -> list[str]:
    """Convert Piper phoneme sentences into lowercase ASCII words.

    Punctuation attached to a word is kept as a trailing mark so the text
    stream retains prosodic boundaries; empty words are dropped.
    """

    words: list[str] = []
    for sentence in sentences:
        joined = "".join(sentence)
        for chunk in joined.split(" "):
            if not chunk:
                continue
            trail = ""
            while chunk and chunk[-1] in _PUNCTUATION:
                trail = _PUNCTUATION[chunk[-1]] + trail
                chunk = chunk[:-1]
            body = _map_word(chunk) if chunk else ""
            if body:
                words.append(body + trail)
            elif trail and words:
                words[-1] += trail
    return words


def romanize(text: str, phonemizer: Phonemizer) -> list[str]:
    """Return the ASCII romanized words for one Persian text."""

    return phonemes_to_words(phonemizer.phonemize(text))


def alignment_letters(word: str) -> str:
    """Strip punctuation so a word is valid input for an a–z forced aligner."""

    letters = "".join(ch for ch in word if "a" <= ch <= "z")
    if not _ASCII_WORD.match(letters):
        raise ValueError(f"word has no alignable letters: {word!r}")
    return letters
