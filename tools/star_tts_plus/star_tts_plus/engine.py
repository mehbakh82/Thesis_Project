"""Lab VITS (star TTS) inference with switchable, measured improvements.

With default options it reproduces the lab Triton service exactly (Persian
digits, ``eng2fa.map_characters``, the service's sentence split, one VITS call
per sentence, 22.05 → 16 kHz ``kaiser_best`` resampling, silence trim, CPU).
Every improvement is an option so that its effect can be measured alone:

* ``cuda`` — run the acoustic model on the GPU;
* ``resample`` — ``"kaiser_best_poly"``: librosa's ``kaiser_best`` (resampy)
  filter, with resampy's own table and index arithmetic, expanded once into a
  polyphase FIR for the fixed 22.05 → 16 kHz ratio and applied with
  ``scipy.signal.resample_poly`` — the same output, several times faster;
* ``numbers`` — spoken reading of times, percentages, decimals, phone numbers,
  ordinals and «هزار»/«صد» (``star_tts_plus.numbers``), applied to the whole
  text before sentence splitting;
* ``noise_scale`` / ``noise_scale_dp`` / ``length_scale`` — VITS sampling and
  speaking-rate settings (None keeps the model's configured values);
* ``fast_espeak`` — in-process espeak-ng with output identical to the fork's
  per-call CLI process (``star_tts_plus.espeak_fast``);
* ``split`` — ``"service"`` or ``"punct"`` (also «؟ ! ؛ :», chunks up to 25
  words), with ``pause_ms`` of silence between chunks;
* ``persian_punct`` — «؟ ؛ ،» → «? ; ,» in each chunk: the fork's punctuation
  set lacks the Persian marks, espeak-ng ends a clause there, and the fork joins
  the clause outputs without a space (two words become one);
* ``tokenizer_fix`` — the fork's number step re-tokenizes every sentence with
  hazm's ``word_tokenize``, whose verb-part joining treats «خواهید/خواهد» as the
  future auxiliary: at the end of a sentence the word is dropped («چه چیزی
  می‌خواهید» is spoken «چه چیزی می»), elsewhere it is glued to the next word
  with «_» (which espeak-ng reads as a space). The fix tokenizes without verb
  joining.

The lab repository is imported, never modified.
"""

from __future__ import annotations

import contextlib
import functools
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 16_000
SOURCE_RATE = 22_050


@dataclass
class Options:
    cuda: bool = False
    resample: str = "kaiser_best"
    numbers: bool = False
    noise_scale: float | None = None
    noise_scale_dp: float | None = None
    length_scale: float | None = None
    split: str = "service"
    pause_ms: float = 0.0
    fast_espeak: bool = False
    persian_punct: bool = False
    tokenizer_fix: bool = False


PERSIAN_PUNCT = str.maketrans({"؟": "?", "؛": ";", "،": ","})


def split_service(text: str) -> list[str]:
    """The lab service's ``_split_sentences``."""

    sents = [s.strip() for s in re.split(r"\.|،", text) if len(s.strip()) > 0]
    normal: list[str] = []
    for s in sents:
        if len(s.split()) < 10:
            normal.append(s)
        else:
            tokens = s.split()
            for i in range(0, len(tokens), 10):
                normal.append(" ".join(tokens[i : i + 10]))
    for i in range(len(normal) - 2, -1, -1):
        if len(normal[i].split() + normal[i + 1].split()) < 10:
            normal[i] = normal[i] + " " + normal.pop(i + 1)
    return normal


def split_punct(text: str, max_words: int = 25) -> list[str]:
    """Split at sentence and clause punctuation, keeping it; cap chunk length."""

    parts = [p.strip() for p in re.split(r"(?<=[.،؟!؛:?])\s+", text) if p.strip()]
    chunks: list[str] = []
    for part in parts:
        tokens = part.split()
        for i in range(0, len(tokens), max_words):
            chunks.append(" ".join(tokens[i : i + max_words]))
    merged: list[str] = []
    for chunk in chunks:  # merge very short fragments into the previous chunk
        if merged and len(chunk.split()) < 3 and len(merged[-1].split()) + len(chunk.split()) <= max_words:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


UP, DOWN = 320, 441  # 16000 / 22050

# Warm-up phrases: windows of 1–12 words over this paragraph give the acoustic
# model many input lengths, so cuDNN has set up its kernels for each before the
# first request (a new input length otherwise costs about 70 ms once on an H100).
WARMUP_TEXT = (
    "سلام، صبح بخیر. امروز هوا کمی سرد است و بهتر است لباس گرم بپوشید. اگر قرص فشار "
    "خونتان را هنوز نخورده‌اید، لطفاً بعد از صبحانه با یک لیوان آب بخورید. برای زانو، "
    "پیاده‌روی آرام روی زمین صاف و ورزش در آب مناسب است. اگر درد شدید شد یا تب داشتید، "
    "حتماً با پزشک خانواده یا اورژانس تماس بگیرید. می‌خواهید برایتان یادآوری بگذارم که "
    "ساعت هشت شب داروی دیگرتان را بخورید؟ من اینجا هستم و هر وقت سؤالی داشتید، با کمال "
    "میل کمک می‌کنم. خواب کافی، غذای سبک شب و نوشیدن آب در طول روز به سلامت شما کمک "
    "می‌کند. نوه‌هایتان هم حتماً خوشحال می‌شوند اگر آخر هفته به آن‌ها زنگ بزنید."
)


def resample(wav: np.ndarray, method: str) -> np.ndarray:
    """22.05 → 16 kHz with librosa (``kaiser_best`` etc.) or ``kaiser_best_poly``."""

    if method == "kaiser_best_poly":
        from scipy.signal import resample_poly

        out = resample_poly(wav, UP, DOWN, window=_kaiser_best_prototype()).astype(np.float32)
        out[int(wav.shape[-1] * float(RATE) / float(SOURCE_RATE)) :] = 0.0  # resampy's length, zero-padded by librosa
        return out
    import librosa

    return librosa.resample(wav, orig_sr=SOURCE_RATE, target_sr=RATE, res_type=method)


@functools.lru_cache(maxsize=1)
def _kaiser_best_prototype() -> np.ndarray:
    """resampy's ``kaiser_best`` downsampling filter on the 1/320-sample grid.

    resampy weights input sample ``n - i`` of an output at fractional input time
    ``n + r/320`` by ``win[o + i * step] + eta * delta[o + i * step]`` with
    ``o + eta = scale * r / 320 * num_table`` and ``step = int(scale * num_table)``
    (and mirrors this for the right wing), using only taps whose index plus
    ``step`` stays inside the table. Distance ``m = 320 * i + r`` on the
    upsampled grid therefore has one weight, and the filter is symmetric.
    """

    from resampy.filters import get_filter

    ratio = RATE / SOURCE_RATE
    table, num_table, _ = get_filter("kaiser_best")  # a stored filter returns its table density here
    num_table = int(num_table)
    win = ratio * table
    delta = np.diff(win, append=win[-1])
    step = int(ratio * num_table)
    taps = (win.size // step) * UP
    m = np.arange(taps)
    index_frac = ratio * (m % UP) / UP * num_table
    offset = index_frac.astype(np.int64)
    index = offset + (m // UP) * step
    valid = index + step <= win.size
    index = np.where(valid, index, 0)
    half = np.where(valid, win[index] + (index_frac - offset) * delta[index], 0.0)
    last = int(np.flatnonzero(half)[-1])
    half = half[: last + 1]
    return np.concatenate([half[:0:-1], half]) / UP  # resample_poly multiplies by UP


@contextlib.contextmanager
def _cwd(path: Path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class StarTTS:
    def __init__(self, repo: Path, options: Options | None = None) -> None:
        self.repo = Path(repo)
        self.options = options or Options()
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        with _cwd(self.repo):  # the heteronym module opens data files by relative path
            from eng2fa import map_characters
            from TTS.tts.utils.text.persian.num2fawords import ordinal_words, words
            from TTS.tts.utils.text.persian.Phonemizer import phonemizer as fa_phonemizer
            from TTS.utils.synthesizer import Synthesizer

            from .numbers import make_normalizer, spoken_style

            if self.options.fast_espeak:
                from TTS.tts.utils.text.phonemizers import espeak_wrapper

                from .espeak_fast import install

                install(espeak_wrapper)

            from hazm import WordTokenizer

            self._phonemizer_module = fa_phonemizer
            self._word_tokenize = (fa_phonemizer.word_tokenize, WordTokenizer(join_verb_parts=False).tokenize)
            self.set_tokenizer_fix(self.options.tokenizer_fix)
            self._normalize = make_normalizer(words, ordinal_words) if self.options.numbers else None
            if self.options.numbers:
                original = fa_phonemizer.Text2Phoneme.convert_number_in_text_to_string
                if not getattr(original, "_star_tts_plus", False):

                    def convert(this, text: str) -> str:  # type: ignore[no-untyped-def]
                        return spoken_style(original(this, text))

                    convert._star_tts_plus = True  # type: ignore[attr-defined]
                    fa_phonemizer.Text2Phoneme.convert_number_in_text_to_string = convert
            self.synthesizer = Synthesizer(
                tts_checkpoint=str(self.repo / "models/model.pth"),
                tts_config_path=str(self.repo / "models/config.json"),
                use_cuda=self.options.cuda,
            )
        self.map_characters = map_characters
        model = self.synthesizer.tts_model
        self._defaults = (model.inference_noise_scale, model.inference_noise_scale_dp, model.length_scale)
        self.configure(self.options.noise_scale, self.options.noise_scale_dp, self.options.length_scale)
        self._digits = str.maketrans("12345678901234567890", "۱۲۳۴۵۶۷۸۹۰١٢٣٤٥٦٧٨٩٠")

    def warmup(self, max_words: int = 12) -> int:
        """Synthesize short windows of ``WARMUP_TEXT``; returns the number of calls.

        Only fills cuDNN's per-shape caches: outputs of later calls are unchanged.
        """

        import io

        tokens = WARMUP_TEXT.split()
        calls = 0
        with contextlib.redirect_stdout(io.StringIO()):
            for size in range(1, max_words + 1):
                for start in range(0, len(tokens) - size + 1, max(1, size)):
                    self.tts(" ".join(tokens[start : start + size]))
                    calls += 1
        return calls

    def set_tokenizer_fix(self, enabled: bool) -> None:
        """Switch the fork's sentence tokenizer (a module global, read per call)."""

        self._phonemizer_module.word_tokenize = self._word_tokenize[1 if enabled else 0]
        self.options.tokenizer_fix = enabled

    def configure(
        self, noise_scale: float | None = None, noise_scale_dp: float | None = None, length_scale: float | None = None
    ) -> None:
        """Set the VITS sampling and rate settings (None: the model's configured value)."""

        model = self.synthesizer.tts_model
        defaults = self._defaults
        model.inference_noise_scale = defaults[0] if noise_scale is None else noise_scale
        model.inference_noise_scale_dp = defaults[1] if noise_scale_dp is None else noise_scale_dp
        model.length_scale = defaults[2] if length_scale is None else length_scale
        self.options.noise_scale, self.options.noise_scale_dp, self.options.length_scale = (
            noise_scale, noise_scale_dp, length_scale
        )

    def chunks(self, text: str) -> list[str]:
        # numbers are rewritten on the whole text first: the service splitter cuts
        # at every «.», including the point of a decimal number
        if self._normalize is not None:
            text = self._normalize(text)
        processed = self.map_characters(text.translate(self._digits))
        chunks = split_punct(processed) if self.options.split == "punct" else split_service(processed)
        if self.options.persian_punct:  # after splitting: both splitters look for the Persian marks
            chunks = [chunk.translate(PERSIAN_PUNCT) for chunk in chunks]
        return chunks

    def raw(self, text: str) -> np.ndarray:
        """The model's 22.05 kHz output for ``text`` (chunks and pauses joined)."""

        import torch

        pause: np.ndarray = np.zeros(int(self.options.pause_ms * SOURCE_RATE / 1000), np.float32)
        pieces: list[np.ndarray] = []
        with _cwd(self.repo), torch.inference_mode():
            for i, chunk in enumerate(self.chunks(text)):
                if i and pause.size:
                    pieces.append(pause)
                pieces.append(np.asarray(self.synthesizer.tts(chunk), dtype=np.float32))
        return np.concatenate(pieces) if pieces else np.zeros(0, np.float32)

    def tts(self, text: str) -> np.ndarray:
        """16 kHz speech for ``text``, leading and trailing silence trimmed."""

        import librosa

        wav = resample(self.raw(text), self.options.resample)
        wav, _ = librosa.effects.trim(wav)
        return wav.astype(np.float32)
