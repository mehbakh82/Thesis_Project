from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("count_clarification_requests", ROOT / "scripts/count_clarification_requests.py")
assert spec and spec.loader
ccr = importlib.util.module_from_spec(spec)
sys.modules["count_clarification_requests"] = ccr
spec.loader.exec_module(ccr)


def test_pattern_matches_repeat_requests_only() -> None:
    assert ccr.REPEAT_REQUEST.search("ببخشید، متوجه منظور شما نشدم. می‌شود دوباره بگویید؟")
    assert ccr.REPEAT_REQUEST.search("ممکن است دوباره با صدای واضح‌تر تکرار کنید؟")
    assert not ccr.REPEAT_REQUEST.search("می‌توانید برای ناهار یک کاسه سوپ جو بخورید.")
    assert not ccr.REPEAT_REQUEST.search("اگر دوباره درد گرفت، با پزشک تماس بگیرید.")


def test_counts_use_the_scored_follow_up_turn(tmp_path: Path) -> None:
    scenarios = {"s1": {"scenario": "s1", "timing": "late", "follow_up": "من گوشت نمی‌خورم"},
                 "s2": {"scenario": "s2", "timing": "early", "follow_up": "دخترم شهر دیگه است"}}
    turns = [
        [{"turn": 1, "reply_text": "سلام", "transcript": "x"},
         {"turn": 2, "reply_text": "متوجه منظورتان نشدم.", "transcript": "من گوش نمیخورم"}],
        [{"turn": 1, "reply_text": "سلام", "transcript": "x"},
         {"turn": 2, "reply_text": "متوجه نشدم", "transcript": "y", "speculation_cancelled": True},
         {"turn": 3, "reply_text": "دوری سخت است.", "transcript": "دخترم شهر دیگه است"}],
    ]
    telemetry, runs, offset = b"", [], 0
    for sid, block in zip(("s1", "s2"), turns, strict=True):
        data = b"".join(json.dumps(t, ensure_ascii=False).encode("utf-8") + b"\n" for t in block)
        telemetry += data
        runs.append({"scenario": sid, "condition": "interrupt", "offset_before": offset, "offset_after": offset + len(data)})
        offset += len(data)
    (tmp_path / "telemetry.jsonl").write_bytes(telemetry)
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")
    rows = ccr.follow_ups(tmp_path, scenarios)
    assert [r["asks_to_repeat"] for r in rows] == [True, False]  # the cancelled speculation is skipped
    table = ccr.summary(rows)
    assert table["interrupt"]["asks_to_repeat"] == 1
    assert table["interrupt"]["by_timing"]["late"] == {"replies": 1, "asks_to_repeat": 1}
    assert table["control"]["replies"] == 0
    assert table["follow_up_cer_median_when_asking"] > table["follow_up_cer_median_otherwise"] == 0
