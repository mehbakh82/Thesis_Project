import os
import sys
from pathlib import Path

import pytest

REPO = Path(os.environ.get("STAR_TTS_REPO", "../star.text_to_speech")).resolve()
if not (REPO / "TTS").is_dir():
    pytest.skip("set STAR_TTS_REPO to the lab TTS repository", allow_module_level=True)
sys.path.insert(0, str(REPO))

_cwd = os.getcwd()
os.chdir(REPO)  # the lab's Persian package opens its data files by relative path
from hazm import WordTokenizer  # noqa: E402
from TTS.tts.utils.text.persian.Phonemizer import phonemizer  # noqa: E402

os.chdir(_cwd)
convert = phonemizer.Text2Phoneme.convert_number_in_text_to_string  # uses the module's word_tokenize


@pytest.fixture()
def fixed_tokenizer():
    original = phonemizer.word_tokenize
    phonemizer.word_tokenize = WordTokenizer(join_verb_parts=False).tokenize
    yield
    phonemizer.word_tokenize = original


def test_fork_drops_a_final_khahid() -> None:
    # the Kasre step leaves the ZWNJ as its own token: «می ‌ خواهید»
    assert convert(None, "چه چیزی می \u200c خواهید") == "چه چیزی می \u200c"


@pytest.mark.usefixtures("fixed_tokenizer")
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("چه چیزی می \u200c خواهید", "چه چیزی می \u200c خواهید"),
        ("می \u200c خواهید چه چیزی", "می \u200c خواهید چه چیزی"),
        ("شما نمی \u200c خواهد", "شما نمی \u200c خواهد"),
        ("او ۲ بار رفته است .", "او دو بار رفته است ."),
    ],
)
def test_fixed_tokenizer_keeps_every_word(text: str, expected: str) -> None:
    assert convert(None, text) == expected
