"""Deterministic word/character error rates for Persian speech evidence."""

from __future__ import annotations

from collections.abc import Sequence


def tokenize(text: str) -> list[str]:
    return [t for t in text.replace("\u200c", " ").split() if t]


def character_tokens(text: str) -> list[str]:
    """Return Unicode code points, excluding whitespace and Persian ZWNJ."""

    return [character for character in text if not character.isspace() and character != "\u200c"]


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Return unit-cost Levenshtein distance for two token sequences."""

    r, h = list(reference), list(hypothesis)
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    """Return Levenshtein error rate, with an explicit empty-reference rule."""

    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def wer(ref: str, hyp: str) -> float:
    return error_rate(tokenize(ref), tokenize(hyp))


def cer(ref: str, hyp: str) -> float:
    return error_rate(character_tokens(ref), character_tokens(hyp))
