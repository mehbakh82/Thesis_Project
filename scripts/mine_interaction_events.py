#!/usr/bin/env python3
"""Mine candidate full-duplex interaction events from diarized conversation windows.

Three candidate kinds are extracted per window from the diarizer's speaker
turns, without consulting any interruption label:

* ``overlap_onset`` – a different speaker starts while the current speaker is
  still talking (possible interruption or backchannel);
* ``short_response`` – a different speaker produces a short turn (<= 1.5 s)
  during or right after the current speaker's turn (possible backchannel);
* ``turn_change`` – a different speaker starts 0.2–2.0 s after the current
  speaker stops (an ordinary, non-overlapping turn exchange).

Candidates are only *proposals*: their class is assigned later by independent
labeling (Gemini) on the audio itself, so labels do not inherit these rules.
Each event records a clip window of ``pre`` seconds before and ``post`` seconds
after the onset, the session, channel, and a deterministic event id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRE_SECONDS = 2.5
POST_SECONDS = 2.0
SHORT_RESPONSE_MAX = 1.5
TURN_GAP = (0.2, 2.0)
MIN_PRIOR_SPEECH = 1.0
MIN_SPACING = 3.0


def _turns(window: dict[str, Any]) -> list[tuple[float, float, str]]:
    turns = [
        (float(t["start"]), float(t["end"]), str(t["speaker"]))
        for t in window.get("speaker_turns") or []
        if float(t["end"]) > float(t["start"])
    ]
    return sorted(turns)


def mine_window(window: dict[str, Any], limit_per_kind: int) -> list[dict[str, Any]]:
    turns = _turns(window)
    duration = float(window["duration"])
    events: list[dict[str, Any]] = []
    kinds: Counter[str] = Counter()
    last_onset = -1e9
    for index, (start, end, speaker) in enumerate(turns):
        if start - PRE_SECONDS < 0 or start + POST_SECONDS > duration:
            continue
        if start - last_onset < MIN_SPACING:
            continue
        others_before = [t for t in turns[:index] if t[2] != speaker and t[1] > start - PRE_SECONDS]
        if not others_before:
            continue
        prior = max(others_before, key=lambda t: t[1])
        prior_speech = min(prior[1], start) - max(prior[0], start - PRE_SECONDS)
        if prior_speech < MIN_PRIOR_SPEECH:
            continue
        own_duration = end - start
        if prior[1] > start + 0.15:
            kind = "short_response" if own_duration <= SHORT_RESPONSE_MAX else "overlap_onset"
        elif TURN_GAP[0] <= start - prior[1] <= TURN_GAP[1]:
            kind = "short_response" if own_duration <= SHORT_RESPONSE_MAX else "turn_change"
        else:
            continue
        if kinds[kind] >= limit_per_kind:
            continue
        kinds[kind] += 1
        last_onset = start
        key = f"{window['window_id']}|{start:.3f}|{speaker}"
        events.append(
            {
                "event_id": "ev-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20],
                "candidate_kind": kind,
                "window_id": window["window_id"],
                "session_id": window["session_id"],
                "channel": window.get("channel"),
                "audio_filepath": window["audio_filepath"],
                "onset_seconds": round(start, 3),
                "clip_start": round(start - PRE_SECONDS, 3),
                "clip_end": round(start + POST_SECONDS, 3),
                "onset_in_clip": PRE_SECONDS,
                "new_speaker_turn_seconds": round(own_duration, 3),
                "prior_speaker_end_offset": round(prior[1] - start, 3),
                "noise_condition": window.get("noise_condition"),
            }
        )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-per-kind", type=int, default=12)
    parser.add_argument(
        "--require-local-audio",
        action="store_true",
        help="skip windows whose WAV is not present locally",
    )
    args = parser.parse_args()

    events: list[dict[str, Any]] = []
    missing = 0
    for manifest in args.windows:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            window = json.loads(line)
            if not (ROOT / window["audio_filepath"]).is_file():
                missing += 1
                if args.require_local_audio:
                    continue
            events.extend(mine_window(window, args.limit_per_kind))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events), encoding="utf-8"
    )
    summary = {
        "events": len(events),
        "windows_without_local_audio": missing,
        "kinds": Counter(e["candidate_kind"] for e in events),
        "sessions": len({e["session_id"] for e in events}),
        "channels": Counter(e["channel"] for e in events),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
