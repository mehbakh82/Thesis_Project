"""Persian listener-token lexicon and self-echo matching for barge-in confirm.

A *backchannel* is operationally a short turn whose transcript consists only of
listener tokens from :data:`BACKCHANNEL_LEXICON`. The same definition is used to
mine evaluation backchannels and by the controller's lexical confirm stage; the
evaluation keeps them independent by transcribing with different ASR models.
"""

from __future__ import annotations

from collections.abc import Iterable

from thesis_s2s.eval.asr_normalize import normalize_for_scoring

BACKCHANNEL_LEXICON = frozenset(
    {
        "آره", "اره", "آرہ", "بله", "بلی", "اوهوم", "اوهم", "هوم", "آها", "اها", "آهان", "اهان",
        "باشه", "درسته", "دقیقا", "دقیقاً", "عالیه", "خب", "خوبه", "آفرین", "جدی", "واقعا", "نه",
        "حتما", "حتماً", "صحیح", "اوکی", "اهوم", "ممنون", "مرسی", "هه", "ههه", "آ", "ا", "اوه",
    }
)


def _close(a: str, b: str) -> bool:
    """Equal, or one edit apart for words of four or more letters."""

    if a == b:
        return True
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b, strict=True)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    return any(long_[:i] + long_[i + 1 :] == short for i in range(len(long_)))


def echo_vocabulary(texts: Iterable[str]) -> set[str]:
    """Normalized words of the assistant text currently being played."""

    words: set[str] = set()
    for text in texts:
        words.update(normalize_for_scoring(text).split())
    return words


def content_tokens(transcript: str, echo_words: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Transcript tokens that are neither listener tokens nor the assistant's own words."""

    tokens = normalize_for_scoring(transcript).split()
    out = []
    for token in tokens:
        if token in BACKCHANNEL_LEXICON:
            continue
        if token in echo_words or any(_close(token, word) for word in echo_words):
            continue
        out.append(token)
    return out


def listener_tokens_only(transcript: str) -> bool:
    tokens = normalize_for_scoring(transcript).split()
    return bool(tokens) and all(token in BACKCHANNEL_LEXICON for token in tokens)
