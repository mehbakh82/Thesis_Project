"""Spoken-Persian reading of numbers for the lab VITS front-end.

Runs on the text *before* the lab front-end's own integer expansion
(``convert_number_in_text_to_string``) and rewrites the patterns it reads
wrongly or unnaturally:

* thousands separators ``500,000`` / ``۵۰۰٬۰۰۰`` → «پانصد هزار» (the lab
  front-end dropped every group after the separator: «پانصد»);
* clock times ``12:45`` → «دوازده و چهل و پنج دقیقه» (``:00`` → just the hour,
  ``:30`` → «… و نیم»);
* dates ``1403/05/12`` (or day first) → «دوازدهم مرداد هزار و چهارصد و سه»,
  with Solar Hijri month names for years 1200–1599 and Gregorian ones for
  1800–2199;
* percentages ``25٪`` / ``25%`` → «بیست و پنج درصد»;
* phone numbers and other long digit strings (a leading zero, or 8+ digits,
  dashes allowed after a leading zero) → digit by digit, in groups, instead of
  «نه میلیارد …»;
* decimals ``2.5`` / ``۲٫۵`` → «دو و پنج دهم»;
* slashes (read «اسلش» before): blood pressure and other ratios ``120/80`` →
  «صد و بیست روی هشتاد» (also any ``a/b`` after «فشار»), simple fractions
  ``1/2`` … ``3/4`` → «یک دوم», and otherwise the Persian decimal slash
  ``37/5`` → «سی و هفت و پنج دهم»;
* ranges ``10-15`` → «ده تا پانزده»;
* ordinals ``3 ام`` / ``۳ام`` → «سوم» (and «سی‌اُم», not the lab's «سیم»);
* Latin units after a number (``5mg``, ``2 kg``, ``37°C``) → Persian words;

and afterwards the formal «یک هزار» / «یکصد» at the start of a number become the
spoken «هزار» / «صد». Number words use the lab repository's ``num2fawords`` and
its «ُ» joiner, so the output matches what the model was trained on.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_FA = "۰۱۲۳۴۵۶۷۸۹"
_AR = "٠١٢٣٤٥٦٧٨٩"
_TO_ASCII = str.maketrans(_FA + _AR, "0123456789" * 2)
_DIGIT_WORDS = ["صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]

D = r"[0-9۰-۹٠-٩]"
TIME_RE = re.compile(rf"(?<!{D})({D}{{1,2}})\s*:\s*({D}{{2}})(?!{D})")
PERCENT_RE = re.compile(rf"({D}+(?:[.٫/]{D}+)?)\s*[٪%]")
DECIMAL_RE = re.compile(rf"(?<!{D})({D}+)[.٫]({D}+)(?!{D})")
LONG_RE = re.compile(rf"(?<!{D})(0{D}{{6,}}|۰{D}{{6,}}|٠{D}{{6,}}|{D}{{8,}})(?!{D})")
ORDINAL_RE = re.compile(rf"(?<!{D})({D}+)\s*‌?(?:اُم|ام)(?![؀-ۿ])")
THOUSANDS_RE = re.compile(rf"(?<![{D[1:-1]},٬.٫])({D}{{1,3}}(?:[,٬]{D}{{3}})+)(?![{D[1:-1]}]|[,٬.٫]{D})")
PHONE_DASH_RE = re.compile(rf"(?<![{D[1:-1]}\-/])([0۰٠]{D}{{1,4}}(?:-{D}{{2,8}})+)(?![{D[1:-1]}\-/])")
DATE_YMD_RE = re.compile(rf"(?<![{D[1:-1]}/])({D}{{4}})[/-]({D}{{1,2}})[/-]({D}{{1,2}})(?![{D[1:-1]}/])")
DATE_DMY_RE = re.compile(rf"(?<![{D[1:-1]}/])({D}{{1,2}})[/-]({D}{{1,2}})[/-]({D}{{4}})(?![{D[1:-1]}/])")
SLASH_RE = re.compile(rf"(?<![{D[1:-1]}/])({D}{{1,3}})\s*/\s*({D}{{1,3}})(?![{D[1:-1]}/])")
RANGE_RE = re.compile(rf"(?<![{D[1:-1]}\-/:.٫])({D}{{1,4}})\s*[-–—]\s*({D}{{1,4}})(?![{D[1:-1]}\-/:])")
UNIT_RE = re.compile(rf"({D})\s*(mcg|µg|mg|kg|g|ml|cc|cm|mm|km|°C|°)(?![A-Za-z])", re.IGNORECASE)
UNITS = {
    "mcg": "میکروگرم", "µg": "میکروگرم", "mg": "میلی‌گرم", "kg": "کیلوگرم", "g": "گرم",
    "ml": "میلی‌لیتر", "cc": "سی‌سی", "cm": "سانتی‌متر", "mm": "میلی‌متر", "km": "کیلومتر",
    "°c": "درجه", "°": "درجه",
}
HIJRI_MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور", "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
GREGORIAN_MONTHS = ["ژانویه", "فوریه", "مارس", "آوریل", "مه", "ژوئن", "ژوئیه", "اوت", "سپتامبر", "اکتبر", "نوامبر", "دسامبر"]


def _ascii(number: str) -> str:
    return number.translate(_TO_ASCII).replace("٫", ".")


def spoken_style(text: str) -> str:
    """Formal «یک هزار…» / «یکصد…» → spoken «هزار…» / «صد…» at a number's start."""

    text = re.sub(r"(^|[\s(«])یک هزار", r"\1هزار", text)
    text = re.sub(r"(^|[\s(«])یکصد", r"\1صد", text)
    return text


def make_normalizer(words: Callable[..., str], ordinal_words: Callable[..., str]) -> Callable[[str], str]:
    """Build the rewriting function around the lab repository's num2fawords."""

    def cardinal(number: str) -> str:
        return spoken_style(words(_ascii(number)))

    def ordinal_of(number: int) -> str:
        base = spoken_style(words(number))
        if base.endswith("سی"):  # the lab gives «سیم» for 30
            return f"{base}‌اُم"
        return spoken_style(ordinal_words(number))

    def thousands_words(match: re.Match[str]) -> str:
        return cardinal(re.sub("[,٬]", "", match.group(1)))

    def date_words(year: str, month: str, day: str, original: str) -> str:
        y, m, d = int(_ascii(year)), int(_ascii(month)), int(_ascii(day))
        months = HIJRI_MONTHS if 1200 <= y < 1600 else GREGORIAN_MONTHS if 1800 <= y < 2200 else None
        if months is None or not (1 <= m <= 12 and 1 <= d <= 31):
            return original
        return f"{ordinal_of(d)} {months[m - 1]} {spoken_style(words(y))}"

    def slash_words(match: re.Match[str]) -> str:
        a, b = match.group(1), match.group(2)
        x, y = int(_ascii(a)), int(_ascii(b))
        before = match.string[max(0, match.start() - 25) : match.start()]
        if "فشار" in before or (len(a) >= 2 and len(b) >= 2 and x >= y):
            return f"{cardinal(a)} روی {cardinal(b)}"
        if 0 < x < y <= 4:
            return f"{cardinal(a)} {ordinal_of(y)}"
        return cardinal(f"{a}.{b}")

    def time_words(match: re.Match[str]) -> str:
        hour, minute = int(_ascii(match.group(1))), int(_ascii(match.group(2)))
        if minute == 0:
            return spoken_style(words(hour))
        if minute == 30:
            return f"{spoken_style(words(hour))} و نیم"
        return f"{spoken_style(words(hour))} و {spoken_style(words(minute))} دقیقه"

    def percent_words(match: re.Match[str]) -> str:
        return f"{cardinal(match.group(1).replace('/', '.'))} درصد"

    def decimal_words(match: re.Match[str]) -> str:
        return cardinal(f"{match.group(1)}.{match.group(2)}")

    def long_words(match: re.Match[str]) -> str:
        digits = _ascii(match.group(1))
        groups = [digits[i : i + 3] for i in range(0, len(digits), 3)]
        if len(groups) > 1 and len(groups[-1]) == 1:  # avoid a lone final digit
            groups[-2:] = [groups[-2] + groups[-1]]
        return "، ".join(" ".join(_DIGIT_WORDS[int(c)] for c in g) for g in groups)

    def ordinal(match: re.Match[str]) -> str:
        return ordinal_of(int(_ascii(match.group(1))))

    def normalize(text: str) -> str:
        text = THOUSANDS_RE.sub(thousands_words, text)
        text = DATE_YMD_RE.sub(lambda m: date_words(m.group(1), m.group(2), m.group(3), m.group(0)), text)
        text = DATE_DMY_RE.sub(lambda m: date_words(m.group(3), m.group(2), m.group(1), m.group(0)), text)
        text = PHONE_DASH_RE.sub(lambda m: m.group(1).replace("-", ""), text)
        text = TIME_RE.sub(time_words, text)
        text = PERCENT_RE.sub(percent_words, text)
        text = LONG_RE.sub(long_words, text)
        text = DECIMAL_RE.sub(decimal_words, text)
        text = SLASH_RE.sub(slash_words, text)
        text = RANGE_RE.sub(lambda m: f"{m.group(1)} تا {m.group(2)}", text)
        text = ORDINAL_RE.sub(ordinal, text)
        text = UNIT_RE.sub(lambda m: f"{m.group(1)} {UNITS[m.group(2).lower()]}", text)
        return text

    return normalize
