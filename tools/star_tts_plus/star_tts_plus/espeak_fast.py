"""In-process espeak-ng phonemization, equivalent to the lab fork's CLI call.

The lab fork runs ``espeak-ng -q -b 1 -v fa --ipa=1 --stdin`` in a new process
for every sentence (about 20 ms). This module calls the same libespeak-ng with
the same flags the CLI uses — UTF-8 input, phoneme mode with IPA and the
``--ipa=N`` separator (1: ``_``; 2: U+0361 tie; 3: zero-width joiner tie), end
pause — and the same phoneme-trace mechanism, writing into an in-memory stream,
so the returned lines are the CLI's lines. ``install`` replaces the fork's
``_espeak_exe`` for exactly ``-v <voice> --ipa=N --stdin`` and falls back to the
original function for anything else.

The fork also constructs a new ``ESpeak`` object per sentence, and each
construction runs ``espeak-ng --voices`` and ``espeak-ng --version`` in
subprocesses (about 14 ms); ``install`` caches both answers, which cannot change
while the process runs.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import threading
from collections.abc import Callable
from typing import Any

AUDIO_OUTPUT_SYNCHRONOUS = 2
POS_CHARACTER = 1
espeakCHARS_UTF8 = 1
espeakPHONEMES = 0x100
espeakENDPAUSE = 0x1000
espeakPHONEMES_IPA = 0x02
espeakPHONEMES_TIE = 0x80
ZWJ = 0x200D

_SYNTH_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)


class InProcessEspeak:
    def __init__(self, voice: str = "fa", ipa: int = 1) -> None:
        self.lib = ctypes.CDLL(ctypes.util.find_library("espeak-ng") or "libespeak-ng.so.1")
        self.libc = ctypes.CDLL(ctypes.util.find_library("c"))
        self.libc.open_memstream.restype = ctypes.c_void_p
        self.libc.open_memstream.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)]
        self.libc.fflush.argtypes = [ctypes.c_void_p]
        self.libc.fclose.argtypes = [ctypes.c_void_p]
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        if self.lib.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0) <= 0:
            raise RuntimeError("espeak_Initialize failed")
        self._callback = _SYNTH_CALLBACK(lambda wav, n, events: 0)  # discard audio, keep phonemes
        self.lib.espeak_SetSynthCallback(self._callback)
        if self.lib.espeak_SetVoiceByName(voice.encode()) != 0:
            raise RuntimeError(f"espeak voice {voice!r} not found")
        self.lib.espeak_SetPhonemeTrace.argtypes = [ctypes.c_int, ctypes.c_void_p]
        self.lib.espeak_Synth.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
        ]
        separator, tie = {1: (ord("_"), 0), 2: (0x0361, espeakPHONEMES_TIE), 3: (ZWJ, espeakPHONEMES_TIE)}[ipa]
        self.options = espeakPHONEMES_IPA | tie | (separator << 8)
        self.flags = espeakCHARS_UTF8 | espeakPHONEMES | espeakENDPAUSE
        self.lock = threading.Lock()

    def lines(self, text: str) -> list[bytes]:
        """Phoneme-trace lines for ``text``, as the CLI produces them.

        The fork writes ``text`` plus a newline to ``espeak-ng --stdin``; the
        CLI's bulk stdin mode overwrites the last byte read (that newline) with
        the terminating NUL, so the text is synthesized without it.
        """

        if not text.endswith("\n"):
            text += "\n"
        data = text.encode("utf-8")[:-1]
        buffer = ctypes.create_string_buffer(data, len(data) + 1)
        out = ctypes.c_char_p()
        size = ctypes.c_size_t()
        with self.lock:
            stream = self.libc.open_memstream(ctypes.byref(out), ctypes.byref(size))
            self.lib.espeak_SetPhonemeTrace(self.options, stream)
            self.lib.espeak_Synth(buffer, len(data) + 1, 0, POS_CHARACTER, 0, self.flags, None, None)
            self.lib.espeak_Synchronize()
            self.libc.fflush(stream)
            raw = ctypes.string_at(out, size.value)
            self.lib.espeak_SetPhonemeTrace(self.options, None)
            self.libc.fclose(stream)
            self.libc.free(out)
        return raw.splitlines(keepends=True)


def install(espeak_wrapper: Any, voice: str = "fa") -> Callable[..., list[bytes]]:
    """Route the fork's phonemizer calls in-process and cache its version and voice queries."""

    engines: dict[int, InProcessEspeak] = {}
    original = espeak_wrapper._espeak_exe
    patterns = {i: ["-v", voice, f"--ipa={i}", "--stdin"] for i in (1, 2, 3)}

    def _espeak_exe(espeak_lib: str, args: list, sync: bool = False, text_input: str | None = None):  # type: ignore[no-untyped-def]
        for ipa, pattern in patterns.items():
            if sync and text_input is not None and list(args) == pattern:
                if ipa not in engines:
                    engines[ipa] = InProcessEspeak(voice, ipa)
                return engines[ipa].lines(text_input)
        return original(espeak_lib, args, sync=sync, text_input=text_input)

    espeak_wrapper._espeak_exe = _espeak_exe
    for name in ("get_espeakng_version", "get_espeak_version"):
        function = getattr(espeak_wrapper, name)
        if not hasattr(function, "cache_info"):
            setattr(espeak_wrapper, name, functools.cache(function))
    languages = espeak_wrapper.ESpeak.supported_languages
    if not hasattr(languages, "cache_info"):
        espeak_wrapper.ESpeak.supported_languages = staticmethod(functools.cache(languages))
    return original
