#!/usr/bin/env python3
"""Input-only construction of deterministic complete-prompt Moshi panels."""

from __future__ import annotations

import sys
import wave
from array import array
from pathlib import Path
from typing import Any


def user_audio_bounds_seconds(path: Path) -> tuple[float, float]:
    """Return the first/last non-zero user-channel PCM positions in a stereo WAV."""

    path = path.resolve()
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        payload = handle.readframes(frames)
    if channels != 2 or sample_width != 2 or sample_rate <= 0 or frames <= 0:
        raise ValueError(f"expected nonempty stereo PCM16 WAV: {path}")
    samples = array("h")
    samples.frombytes(payload)
    if sys.byteorder == "big":
        samples.byteswap()
    if len(samples) != frames * channels:
        raise ValueError(f"unexpected PCM payload length: {path}")
    if samples.itemsize != sample_width:
        raise ValueError(f"host PCM16 representation is incompatible: {path}")
    first_user_sample = next(
        (index for index in range(1, len(samples), 2) if samples[index] != 0),
        None,
    )
    final_user_sample = next(
        (index for index in reversed(range(1, len(samples), 2)) if samples[index] != 0),
        None,
    )
    if first_user_sample is None or final_user_sample is None:
        raise ValueError(f"user channel is empty: {path}")
    first_frame = first_user_sample // channels
    final_frame_exclusive = final_user_sample // channels + 1
    return first_frame / sample_rate, final_frame_exclusive / sample_rate


def evenly_spaced_members(values: list[int], panel_size: int = 9) -> tuple[int, ...]:
    """Select manifest-ordered members at deterministic floor-spaced ranks."""

    if panel_size < 2 or len(values) < panel_size:
        raise ValueError("values must cover a panel of at least two members")
    return tuple(
        values[index * (len(values) - 1) // (panel_size - 1)]
        for index in range(panel_size)
    )


def complete_prompt_panel(
    rows: list[dict[str, Any]],
    *,
    input_seconds: float,
    panel_size: int = 9,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Derive a panel using only prompts whose complete user audio is streamed."""

    if input_seconds <= 0:
        raise ValueError("input_seconds must be positive")
    eligible: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        raw_path = row.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"manifest row {index} has no audio path")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        user_start, user_end = user_audio_bounds_seconds(path)
        if user_end <= input_seconds:
            eligible.append(
                {
                    "manifest_index": index,
                    "path": path,
                    "user_audio_start_seconds": user_start,
                    "user_audio_end_seconds": user_end,
                }
            )
    selected_indices = evenly_spaced_members(
        [int(row["manifest_index"]) for row in eligible],
        panel_size=panel_size,
    )
    return selected_indices, eligible
