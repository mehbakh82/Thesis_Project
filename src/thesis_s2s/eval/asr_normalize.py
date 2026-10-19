"""Persian ASR scoring normalization (punctuation-free, orthography-unified).

This mirrors the normalizer of the user's audited Persian ASR evaluation
suite: Arabic-script variants are mapped to Persian letters, diacritics and
tatweel are removed, Persian/Arabic digits become ASCII digits, ZWNJ/ZWJ are
removed (so half-spaced and joined spellings match), and punctuation is
replaced by spaces. Unlike the conservative ``fa-verbatim-2`` text normalizer,
it never lets punctuation count as a word or character error, so systems
that punctuate are not penalized against unpunctuated references.
"""

from __future__ import annotations

import re

from thesis_s2s.eval.wer import edit_distance

ARABIC_TO_PERSIAN = str.maketrans(
    {"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "ؤ": "و", "إ": "ا", "أ": "ا", "ئ": "ی"}
)
DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
DIACRITICS = re.compile(r"[ً-ٰٟ]")
PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
SPACES = re.compile(r"\s+")
NORMALIZER_VERSION = "fa-asr-score-v1"


def normalize_for_scoring(text: str) -> str:
    value = DIACRITICS.sub("", text).replace("ـ", "")
    value = value.translate(ARABIC_TO_PERSIAN).translate(DIGITS)
    value = value.replace("‌", "").replace("‍", "")
    value = PUNCT.sub(" ", value).replace("_", " ")
    return SPACES.sub(" ", value).strip().lower()


def error_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Word and character edit counts after scoring normalization."""

    ref = normalize_for_scoring(reference)
    hyp = normalize_for_scoring(hypothesis)
    ref_words, hyp_words = ref.split(), hyp.split()
    ref_chars = [c for c in ref if not c.isspace()]
    hyp_chars = [c for c in hyp if not c.isspace()]
    return {
        "word_errors": edit_distance(ref_words, hyp_words),
        "reference_words": len(ref_words),
        "char_errors": edit_distance(ref_chars, hyp_chars),
        "reference_chars": len(ref_chars),
    }
