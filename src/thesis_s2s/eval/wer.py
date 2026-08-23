"""Simple WER for S2S replies vs intended Persian text."""

from __future__ import annotations


def tokenize(text: str) -> list[str]:
    return [t for t in text.replace("\u200c", " ").split() if t]


def wer(ref: str, hyp: str) -> float:
    r, h = tokenize(ref), tokenize(hyp)
    if not r:
        return 0.0 if not h else 1.0
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
    return dp[n][m] / n
