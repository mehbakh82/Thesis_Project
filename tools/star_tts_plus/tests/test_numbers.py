import os
import sys
from pathlib import Path

import pytest

REPO = Path(os.environ.get("STAR_TTS_REPO", "../star.text_to_speech")).resolve()
if not (REPO / "TTS").is_dir():
    pytest.skip("set STAR_TTS_REPO to the lab TTS repository", allow_module_level=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_cwd = os.getcwd()
os.chdir(REPO)  # the lab's Persian package opens its data files by relative path
from TTS.tts.utils.text.persian.num2fawords import ordinal_words, words  # noqa: E402

os.chdir(_cwd)

from star_tts_plus.numbers import make_normalizer, spoken_style  # noqa: E402

normalize = make_normalizer(words, ordinal_words)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ساعت ۱۲:۴۵ است", "ساعت دوازده و چهلُ پنج دقیقه است"),
        ("ساعت 8:00 صبح", "ساعت هشت صبح"),
        ("ساعت ۲:۳۰ بعدازظهر", "ساعت دو و نیم بعدازظهر"),
        ("تخفیف ۲۵٪ است", "تخفیف بیستُ پنج درصد است"),
        ("دمای ۲.۵ درجه", "دمای دوُ پنج دهم درجه"),
        ("روز ۳ ام ماه", "روز سوم ماه"),
        ("شماره ۰۹۱۲۳۴۵۶۷۸۹ است", "شماره صفر نه یک، دو سه چهار، پنج شش هفت، هشت نه است"),
        ("کد ۱۲۳۴۵۶۷۸ را بزنید", "کد یک دو سه، چهار پنج شش، هفت هشت را بزنید"),
        ("سال ۱۴۰۳", "سال ۱۴۰۳"),  # plain integers are left to the lab front-end
        ("قیمت ۵۰۰,۰۰۰ تومان", "قیمت پانصد هزار تومان"),
        ("قیمت ۲۱٬۵۰۰٬۰۰۰ تومان", "قیمت بیستُ یک میلیونُ پانصد هزار تومان"),
        ("اعداد ۱, ۲, ۳", "اعداد ۱, ۲, ۳"),  # a list, not thousands
        ("تاریخ ۱۴۰۳/۰۵/۱۲ است", "تاریخ دوازدهم مرداد هزارُ چهارصدُ سه است"),
        ("روز ۳۰/۶/۱۴۰۲", "روز سی‌اُم شهریور هزارُ چهارصدُ دو"),
        ("on 2024-03-01", "on یکم مارس دو هزارُ بیستُ چهار"),
        ("فشار خون ۱۲۰/۸۰ است", "فشار خون صدُ بیست روی هشتاد است"),
        ("فشار ۱۲/۸", "فشار دوازده روی هشت"),
        ("۱/۲ قرص", "یک دوم قرص"),
        ("دمای ۳۷/۵ درجه", "دمای سیُ هفتُ پنج دهم درجه"),
        ("۱۰-۱۵ دقیقه", "۱۰ تا ۱۵ دقیقه"),
        ("تلفن ۰۲۱-۸۸۸۸۸۸۸۸", "تلفن صفر دو یک، هشت هشت هشت، هشت هشت هشت، هشت هشت"),
        ("۵mg دارو", "۵ میلی‌گرم دارو"),
        ("۲ kg برنج", "۲ کیلوگرم برنج"),
        ("تب ۳۸°C", "تب ۳۸ درجه"),
        ("روز ۳۰ ام", "روز سی‌اُم"),
    ],
)
def test_normalize(text: str, expected: str) -> None:
    assert normalize(text) == expected


@pytest.mark.parametrize(
    ("number", "expected"),
    [(1403, "هزارُ چهارصدُ سه"), (115, "صدُ پانزده"), (1000, "هزار"), (2500, "دو هزارُ پانصد"), (100, "صد")],
)
def test_spoken_style_of_integers(number: int, expected: str) -> None:
    assert spoken_style(words(number)) == expected
